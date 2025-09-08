# ------------------------------------------------------------------------------
# Adversarial Domain Adaptation Training for PIDNet (LoveDA)
# ------------------------------------------------------------------------------

import logging
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import functional as F

from utils.utils import AverageMeter, adjust_learning_rate
from utils.utils import get_confusion_matrix

def train_adversarial(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
                     trainloader_source, trainloader_target,
                     optimizer_G, optimizer_D, model_G, model_D, writer_dict):
    """
    Adversarial domain adaptation training loop (output-space, single-level).
    """
    model_G.train()
    model_D.train()

    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc  = AverageMeter()
    avg_sem_loss = AverageMeter()
    avg_bce_loss = AverageMeter()
    adv_loss_meter = AverageMeter()
    d_loss_meter = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    source_iter = iter(trainloader_source)
    target_iter = iter(trainloader_target)

    lambda_adv = config.LOSS.LAMBDA_ADV

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
        for param in model_D.parameters():
            param.requires_grad = False

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # a) Segmentation loss on source
        losses_s, outputs_s, acc_s, loss_list_s = model_G(images_s, labels_s, bd_gts_s)
        loss_seg = losses_s.mean()
        acc = acc_s.mean()

        # b) Adversarial loss on target
        _, outputs_t, _, _ = model_G(images_t, None, None)
        # outputs_t is a list: [aux, main, bd], use main output
        pred_t = F.softmax(outputs_t[1], dim=1)
        d_out_t = model_D(pred_t)
        # Target domain label = 0 (fake/source=1, target=0)
        adv_loss = lambda_adv * F.binary_cross_entropy_with_logits(
            d_out_t, torch.zeros_like(d_out_t)
        )
        total_loss_G = loss_seg + adv_loss
        total_loss_G.backward()
        optimizer_G.step()

        # === 4. Train Discriminator (D) ===
        for param in model_D.parameters():
            param.requires_grad = True

        optimizer_D.zero_grad()
        # Source: should be classified as 1
        pred_s = F.softmax(outputs_s[1].detach(), dim=1)
        d_out_s = model_D(pred_s)
        loss_D_s = F.binary_cross_entropy_with_logits(
            d_out_s, torch.ones_like(d_out_s)
        )

        # Target: should be classified as 0
        pred_t = F.softmax(outputs_t[1].detach(), dim=1)
        d_out_t = model_D(pred_t)
        loss_D_t = F.binary_cross_entropy_with_logits(
            d_out_t, torch.zeros_like(d_out_t)
        )

        loss_D = 0.5 * (loss_D_s + loss_D_t)
        loss_D.backward()
        optimizer_D.step()

        # === 5. Logging and meters ===
        batch_time.update(time.time() - tic)
        tic = time.time()
        ave_loss.update(loss_seg.item())
        ave_acc.update(acc.item())
        avg_sem_loss.update(loss_list_s[0].mean().item())
        avg_bce_loss.update(loss_list_s[1].mean().item())
        adv_loss_meter.update(adv_loss.item())
        d_loss_meter.update(loss_D.item())

        lr = adjust_learning_rate(optimizer_G, base_lr, num_iters, i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = (
                f'Epoch: [{epoch}/{num_epoch}] Iter:[{i_iter}/{epoch_iters}], '
                f'Time: {batch_time.average():.2f}, lr: {lr}, '
                f'Loss: {ave_loss.average():.6f}, Acc: {ave_acc.average():.6f}, '
                f'Semantic loss: {avg_sem_loss.average():.6f}, '
                f'BCE loss: {avg_bce_loss.average():.6f}, '
                f'Adv loss: {adv_loss_meter.average():.6f}, '
                f'D loss: {d_loss_meter.average():.6f}'
            )
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer.add_scalar('train_adv_loss', adv_loss_meter.average(), global_steps)
    writer.add_scalar('train_d_loss', d_loss_meter.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict):
    # (identical to function.py)
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

            losses, pred, _, _ = model(image, label, bd_gts)
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

if __name__ == '__main__':
    main()
