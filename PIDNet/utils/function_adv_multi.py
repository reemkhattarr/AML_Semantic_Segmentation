import torch
import torch.nn.functional as F
import time
from utils.utils import AverageMeter, adjust_learning_rate, get_confusion_matrix
import logging
from torch.autograd import Variable

def train_adversarial_multi(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
                            trainloader_source, trainloader_target,
                            optimizer_G, optimizer_D_list, model_G, model_D_list, writer_dict):
    """
    Adversarial domain adaptation training loop for multi-level outputs.
    model_D_list: list of discriminators (one per output level)
    optimizer_D_list: list of optimizers (one per discriminator)
    """
    model_G.train()
    for model_D in model_D_list:
        model_D.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    adv_loss_meter_list = [AverageMeter() for _ in model_D_list]
    d_loss_meter_list = [AverageMeter() for _ in model_D_list]

    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    source_iter = iter(trainloader_source)
    target_iter = iter(trainloader_target)
    lambda_adv_list = config.LOSS.LAMBDA_ADV_LIST  # e.g. [0.0002, 0.001, 0.0002]

    for i_iter in range(epoch_iters):
        # === 1. Get source batch (labeled) ===
        try:
            batch_s = next(source_iter)
        except StopIteration:
            source_iter = iter(trainloader_source)
            batch_s = next(source_iter)

        images_s, labels_s, bd_gts_s, _, _ = batch_s
        images_s = images_s.cuda()
        labels_s = labels_s.long().cuda()
        bd_gts_s = bd_gts_s.float().cuda()

        # === 2. Get target batch (unlabeled) ===
        try:
            batch_t = next(target_iter)
        except StopIteration:
            target_iter = iter(trainloader_target)
            batch_t = next(target_iter)
        images_t, _, bd_gts_t, _, _ = batch_t
        images_t = images_t.cuda()

        # === 3. Train Segmentation Model (G) ===
        for param in [p for model_D in model_D_list for p in model_D.parameters()]:
            param.requires_grad = False

        optimizer_G.zero_grad()
        for optimizer_D in optimizer_D_list:
            optimizer_D.zero_grad()

        # a) Segmentation loss on source
        losses_s, outputs_s, acc_s, loss_list_s = model_G(images_s, labels_s, bd_gts_s)
        loss_seg = losses_s.mean()
        acc = acc_s.mean()

        # b) Multi-level adversarial loss on target
        _, outputs_t, _, _ = model_G(images_t, None, None)

        # outputs_t is list of outputs; each passes through corresponding discriminator
        adv_loss_total = 0
        for i, (pred_t, model_D, lambda_adv) in enumerate(zip(outputs_t, model_D_list, lambda_adv_list)):
            pred_t_softmax = F.softmax(pred_t, dim=1)
            d_out_t = model_D(pred_t_softmax)
            adv_loss_i = lambda_adv * F.binary_cross_entropy_with_logits(
                d_out_t, torch.zeros_like(d_out_t))
            adv_loss_total += adv_loss_i
            adv_loss_meter_list[i].update(adv_loss_i.item())

        total_loss_G = loss_seg + adv_loss_total
        total_loss_G.backward()
        optimizer_G.step()
        
        #print('len(outputs_s):', len(outputs_s), 'i:', i)


        # === 4. Train Discriminators (D) ===
        for i, model_D in enumerate(model_D_list):
            for param in model_D.parameters():
                param.requires_grad = True
            optimizer_D_list[i].zero_grad()

            # Source domain (label 1)
            pred_s = F.softmax(outputs_s[i].detach(), dim=1)
            d_out_s = model_D(pred_s)
            loss_D_s = F.binary_cross_entropy_with_logits(
                d_out_s, torch.ones_like(d_out_s))

            # Target domain (label 0)
            pred_t = F.softmax(outputs_t[i].detach(), dim=1)
            d_out_t = model_D(pred_t)
            loss_D_t = F.binary_cross_entropy_with_logits(
                d_out_t, torch.zeros_like(d_out_t))

            loss_D = 0.5 * (loss_D_s + loss_D_t)
            loss_D.backward()
            optimizer_D_list[i].step()
            d_loss_meter_list[i].update(loss_D.item())

        # === 5. Logging ===
        batch_time.update(time.time() - tic)
        tic = time.time()
        ave_loss.update(loss_seg.item())
        ave_acc.update(acc.item())
        lr = adjust_learning_rate(optimizer_G, base_lr, num_iters, i_iter + cur_iters)
        if i_iter % config.PRINT_FREQ == 0:
            adv_loss_msg = ', '.join([f'Adv_loss_{j}: {adv_loss_meter_list[j].average():.6f}' for j in range(len(adv_loss_meter_list))])
            d_loss_msg = ', '.join([f'D_loss_{j}: {d_loss_meter_list[j].average():.6f}' for j in range(len(d_loss_meter_list))])
            msg = (
                f'Epoch: [{epoch}/{num_epoch}] Iter:[{i_iter}/{epoch_iters}], '
                f'Time: {batch_time.average():.2f}, lr: {lr}, '
                f'Loss: {ave_loss.average():.6f}, Acc: {ave_acc.average():.6f}, '
                f'{adv_loss_msg}, {d_loss_msg}'
            )
            logging.info(msg)
            writer.add_scalar('train_loss', ave_loss.average(), global_steps)
            for j in range(len(adv_loss_meter_list)):
                writer.add_scalar(f'train_adv_loss_{j}', adv_loss_meter_list[j].average(), global_steps)
                writer.add_scalar(f'train_d_loss_{j}', d_loss_meter_list[j].average(), global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate_multi(config, testloader, model, writer_dict):
    model.eval()
    ave_loss = AverageMeter()
    nums = config.MODEL.NUM_OUTPUTS  # e.g. 3 for your multi-output model
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES, nums))
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            image, label, bd_gts, _, _ = batch
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            bd_gts = bd_gts.float().cuda()
            losses, pred, _, _ = model(image, label, bd_gts)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for i, x in enumerate(pred):
                x = F.interpolate(
                    input=x, size=size[-2:], mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
                confusion_matrix[..., i] += get_confusion_matrix(
                    label, x, size, config.DATASET.NUM_CLASSES, config.TRAIN.IGNORE_LABEL)

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
            logging.info(f'{i} {IoU_array} {mean_IoU}')

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
        writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU, IoU_array

