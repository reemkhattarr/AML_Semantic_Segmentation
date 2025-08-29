import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils.utils import AverageMeter, adjust_learning_rate, get_confusion_matrix
from utils.dacs_utils import generate_pseudo_labels, classmix, generate_edge_map
import numpy as np
import logging  
import os   

import albumentations as A

# strong augmentation pipeline
strong_aug = A.Compose([
    A.ColorJitter(p=0.8),
    # A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    # A.HorizontalFlip(p=0.5),
])

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denorm_and_augment(img_tensor, aug=strong_aug):
    """
    img_tensor: (C, H, W), normalized (ImageNet)
    Returns: (C, H, W), normalized (ImageNet), after augmentation
    """
    # 1. Denormalize to [0,1]
    mean = torch.tensor(IMAGENET_MEAN, device=img_tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img_tensor.device).view(-1, 1, 1)
    img = img_tensor * std + mean  # [0,1]
    img = img.clamp(0, 1)

    # 2. To numpy, [0,255] uint8
    img_np = (img.cpu().numpy() * 255).astype(np.uint8)
    img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)

    # 3. Apply augmentation
    augmented = aug(image=img_np)
    img_aug = augmented['image']

    # 4. Back to float32 [0,1]
    img_aug = img_aug.astype(np.float32) / 255.0
    img_aug = np.transpose(img_aug, (2, 0, 1))  # (C, H, W)

    # 5. Re-normalize
    img_aug = torch.tensor(img_aug, dtype=torch.float, device=img_tensor.device)
    img_aug = (img_aug - mean) / std

    return img_aug



def update_ema_variables(ema_model, student_model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    for ema_param, param in zip(ema_model.parameters(), student_model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param.data[:] + (1 - alpha_teacher) * param.data[:]


def get_dacs_loss_weight(base_weight, cur_iter, rampup_iter=5000):
    if cur_iter < rampup_iter:
        return base_weight * float(cur_iter) / rampup_iter
    else:
        return base_weight


def train_dacs(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
               dual_loader, optimizer, model, teacher_model, writer_dict,
                pseudo_thr=0.968, dacs_loss_weight=0.1):
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
        teacher_model.eval()
        with torch.no_grad():
            tgt_pseudo_lbl, tgt_weight_mask = generate_pseudo_labels(
                teacher_model, tgt_img, threshold=pseudo_thr
            )
        teacher_model.train()
        
        valid_pixels = (tgt_pseudo_lbl != config.TRAIN.IGNORE_LABEL).sum().item()
        if valid_pixels == 0:
            # Skip this iteration if no valid pixels in pseudo labels
            continue
        
        source_weight = torch.ones_like(src_lbl, dtype=torch.float)

        # 3. DACS mixing (ClassMix)
        mixed_img, mixed_lbl, source_mask, mixed_weight = classmix(
            src_img, src_lbl, tgt_img, tgt_pseudo_lbl,
            source_weight=source_weight,
            target_weight=tgt_weight_mask,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            classmix_frac=config.TRAIN.DACS_CLASSMIX_FRAC
        )
        
        valid_mixed_pixels = (mixed_lbl != config.TRAIN.IGNORE_LABEL).sum().item()
        if valid_mixed_pixels == 0:
            continue  # Skip batch if all pixels are ignore_label
        
        mixed_bd = generate_edge_map(mixed_lbl, mixed_weight, edge_size=3, ignore_label=config.TRAIN.IGNORE_LABEL) 
        if (mixed_bd != config.TRAIN.IGNORE_LABEL).sum().item() == 0:
            print("Warning: All boundary labels are ignore_label after mixing. Skipping batch.")
        mixed_losses, _, _, _ = model(mixed_img, mixed_lbl, mixed_bd)
        mixed_loss = (mixed_losses * mixed_weight).sum() / (mixed_weight.sum() + 1e-6)
        
        cur_dacs_loss_weight = get_dacs_loss_weight(dacs_loss_weight, i_iter + cur_iters)
        loss = loss + cur_dacs_loss_weight * mixed_loss

        # === Consistency Regularization ===
        # a. Apply strong augmentation to target images (on normalized tensors)
        tgt_img_strong = torch.stack([
            denorm_and_augment(img) for img in tgt_img
        ])
        
        # b. Get student prediction on strongly augmented images
        student_logits = model.module.model(tgt_img_strong)
        if isinstance(student_logits, (list, tuple)):
            student_logits = student_logits[-2] if len(student_logits) > 1 else student_logits[0]
        student_logits = F.interpolate(student_logits, size=tgt_img.shape[2:], mode='bilinear', align_corners=False)
        
        # c. Consistency loss (cross-entropy, weighted by confidence)
        consistency_loss = F.cross_entropy(student_logits, tgt_pseudo_lbl, reduction='none', ignore_index=config.TRAIN.IGNORE_LABEL)
        consistency_loss = (consistency_loss * tgt_weight_mask).sum() / (tgt_weight_mask.sum() + 1e-6)
        consistency_weight = min(0.1, (i_iter + cur_iters) / 10000 * 0.1)  # ramp up to 0.1 over 10k iters
        # d. Add to total loss (with a weight, e.g. 1.0 or tune as needed)
        loss = loss + consistency_weight * consistency_loss

        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 4. Update teacher model (EMA)
        # update_ema(model.module.model, teacher_model.module, alpha=0.99)
        
        alpha_teacher = 0.99
        update_ema_variables(teacher_model.module, model.module.model, alpha_teacher, i_iter + cur_iters)

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
        
        #teacher_model.load_state_dict(model.module.model.state_dict())

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
