import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils.utils import AverageMeter, adjust_learning_rate, get_confusion_matrix
from utils.dacs_utils import generate_pseudo_labels, classmix, generate_edge_map
import numpy as np
import logging  # FIXED: Added missing import
import os      # FIXED: Added missing import


'''
def update_ema(student_model, teacher_model, alpha=0.99):
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
'''

def train_dacs(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
               dual_loader, optimizer, model, writer_dict,
               dacs_prob=0.7, pseudo_thr=0.968, dacs_loss_weight=0.1):
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    tic.record()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter in range(epoch_iters):
        (src_img, src_lbl, src_bd, _, _), (tgt_img, _, _, _, _) = next(dual_loader)
        src_img = src_img.cuda()
        src_lbl = src_lbl.long().cuda()
        src_bd = src_bd.float().cuda()
        tgt_img = tgt_img.cuda()

        # 1. Supervised loss on source
        losses, _, acc, loss_list = model(src_img, src_lbl, src_bd)
        loss = losses.mean()
        acc  = acc.mean()

        # 2. Pseudo-label target
        # Ensure model is in eval mode for pseudo-labels
        model.eval()
        tgt_pseudo_lbl = generate_pseudo_labels(model.module.model, tgt_img, threshold=pseudo_thr, ignore_label=config.TRAIN.IGNORE_LABEL)
        model.train()
        
        valid_pixels = (tgt_pseudo_lbl != config.TRAIN.IGNORE_LABEL).sum().item()
        if valid_pixels == 0:
            # Skip this iteration if no valid pixels in pseudo labels
            continue

        # 3. DACS mixing (ClassMix)
        if np.random.rand() < dacs_prob:
            mixed_img, mixed_lbl, source_mask = classmix(
                src_img, src_lbl, tgt_img, tgt_pseudo_lbl,
                ignore_label=config.TRAIN.IGNORE_LABEL,
                classmix_frac=config.TRAIN.DACS_CLASSMIX_FRAC
            )
            
            valid_mixed_pixels = (mixed_lbl != config.TRAIN.IGNORE_LABEL).sum().item()
            if valid_mixed_pixels == 0:
                continue  # Skip batch if all pixels are ignore_label
            
            mixed_bd = generate_edge_map(mixed_lbl, edge_size=3, ignore_label=config.TRAIN.IGNORE_LABEL) 
            if (mixed_bd != config.TRAIN.IGNORE_LABEL).sum().item() == 0:
                print("Warning: All boundary labels are ignore_label after mixing. Skipping batch.")
            mixed_losses, _, _, _ = model(mixed_img, mixed_lbl, mixed_bd)
            mixed_loss = mixed_losses.mean()
            loss = loss + dacs_loss_weight * mixed_loss

        model.zero_grad()
        loss.backward()
        optimizer.step()
        

        # measure elapsed time
        toc.record()
        torch.cuda.synchronize()
        batch_time.update(tic.elapsed_time(toc) / 1000.0)
        tic.record()

        # update average loss
        ave_loss.update(loss.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list[0].mean().item())
        avg_bce_loss.update(loss_list[1].mean().item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            print(f'Epoch: [{epoch}/{num_epoch}] Iter:[{i_iter}/{epoch_iters}], '
                  f'Loss: {ave_loss.average():.6f}, Acc:{ave_acc.average():.6f}, '
                  f'Semantic loss: {avg_sem_loss.average():.6f}, BCE loss: {avg_bce_loss.average():.6f}')

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()

            # Robustly handle model output signature
            model_output = model(image, label, bd_gts)
            if len(model_output) == 4:
                losses, pred, _, _ = model_output
            elif len(model_output) == 2:
                losses, pred = model_output
            else:
                raise RuntimeError("Model output signature not recognized.")

            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )

                confusion_matrix[..., i] += get_confusion_matrix(
                    label,
                    x,
                    size,
                    config.DATASET.NUM_CLASSES,
                    config.TRAIN.IGNORE_LABEL
                )

            if idx % 10 == 0:
                print(idx)

            loss = losses.mean()
            ave_loss.update(loss.item())

    for i in range(nums):
        pos = confusion_matrix[..., i].sum(1)
        res = confusion_matrix[..., i].sum(0)
        tp = np.diag(confusion_matrix[..., i])
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        
        logging.info('{} {} {}'.format(i, IoU_array, mean_IoU))

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array

def testval(config, test_dataset, testloader, model,
            sv_dir='./', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros((config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, _, name = batch
            size = label.size()
            pred = test_dataset.single_scale_inference(config, model, image.cuda())

            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
            
            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir, 'val_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model,
        sv_dir='./', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.single_scale_inference(
                config,
                model,
                image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(
                    pred, size[-2:],
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                
            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
