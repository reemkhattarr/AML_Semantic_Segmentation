# # =============================
# # utils/adv_training.py  (loop di training avversario)
# # =============================
# import logging
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.utils import AverageMeter, adjust_learning_rate
# from configs import config


# # helper per scegliere la testa "main"
# def _pick_main_from_seg_outputs(seg_outputs):
#     # seg_outputs è la parte senza boundary: list/tuple di teste (es. [aux, main])
#     if isinstance(seg_outputs, (list, tuple)):
#         main_idx = getattr(config.TEST, 'OUTPUT_INDEX', len(seg_outputs) - 1)
#         # fallback sicuro se l'indice non esiste
#         if not (0 <= main_idx < len(seg_outputs)):
#             main_idx = len(seg_outputs) - 1
#         return seg_outputs[main_idx]
#     return seg_outputs  # già tensore


# def _freeze(module: nn.Module, flag: bool):
#     for p in module.parameters():
#         p.requires_grad = not flag


# def train_adv(config, epoch, num_epoch, epoch_iters, base_lr,
#               num_iters, trainloader, targetloader, optimizer, optimizer_D,
#               model, D, bce_adv, lambda_adv, writer_dict):
#     """Training adversario per un'epoca.
#     - SOURCE: loss supervisionata (la tua FullModel forward)
#     - TARGET: perdita avversaria (G-step)
#     - D-step: aggiorna il discriminatore con source=1, target=0
#     """
#     model.train()
#     D.train()

#     batch_time = AverageMeter()
#     ave_loss = AverageMeter()
#     ave_acc  = AverageMeter()
#     avg_sem_loss = AverageMeter()
#     avg_bce_loss = AverageMeter()

#     import time
#     tic = time.time()
#     cur_iters = epoch * epoch_iters
#     writer = writer_dict['writer']
#     global_steps = writer_dict['train_global_steps']

#     target_iter = iter(targetloader)

#     for i_iter, batch_s in enumerate(trainloader, 0):
#         # ------- SOURCE supervisionato -------
#         images_s, labels_s, bd_gts_s, _, _ = batch_s
#         images_s = images_s.cuda(non_blocking=True)
#         labels_s = labels_s.long().cuda(non_blocking=True)
#         bd_gts_s = bd_gts_s.float().cuda(non_blocking=True)

#         losses, pred_s, acc_s, loss_list = model(images_s, labels_s, bd_gts_s)
#         loss_sup = losses.mean()
#         acc_s    = acc_s.mean()

#         # pred_s: lista delle teste senza boundary -> prendo la main
#         pred_s_logits = _pick_main_from_seg_outputs(pred_s)


#         # ------- TARGET adversario: G-step -------
#         try:
#             batch_t = next(target_iter)
#         except StopIteration:
#             target_iter = iter(targetloader)
#             batch_t = next(target_iter)

#         images_t, _, _, _, _ = batch_t
#         images_t = images_t.cuda(non_blocking=True)

#         # Disattiva grad su D (G-step)
#         _freeze(D, True)
#         optimizer.zero_grad(set_to_none=True)

#         # forward “puro” del backbone sul target (bypass di FullModel)
#         raw_outs_t = model.module.model(images_t)     # restituisce [aux, main, boundary]
#         if isinstance(raw_outs_t, (list, tuple)):
#             seg_outs_t = raw_outs_t[:-1]             # rimuovi boundary
#         else:
#             seg_outs_t = raw_outs_t                  # (nel tuo caso è una lista)

#         logits_t = _pick_main_from_seg_outputs(seg_outs_t)
#         probs_t  = F.softmax(logits_t, dim=1)
#         pred_D_t = D(probs_t)
#         target_real = torch.ones_like(pred_D_t, device=pred_D_t.device)
#         loss_adv_G = bce_adv(pred_D_t, target_real)

#         total_loss_G = loss_sup + lambda_adv * loss_adv_G
#         total_loss_G.backward()
#         optimizer.step()

#         # ------- D-step -------
#         _freeze(D, False)
#         optimizer_D.zero_grad(set_to_none=True)

#         with torch.no_grad():
#             probs_s = F.softmax(pred_s_logits, dim=1)
#             probs_t_det = probs_t.detach()

#         pred_D_s = D(probs_s.detach())
#         pred_D_t = D(probs_t_det)

#         label_s = torch.ones_like(pred_D_s, device=pred_D_s.device)
#         label_t = torch.zeros_like(pred_D_t, device=pred_D_t.device)

#         loss_D = 0.5 * (bce_adv(pred_D_s, label_s) + bce_adv(pred_D_t, label_t))
#         loss_D.backward()
#         optimizer_D.step()

#         # ------- logging + LR -------
#         batch_time.update(time.time() - tic); tic = time.time()
#         ave_loss.update(total_loss_G.item())
#         ave_acc.update(acc_s.item())
#         # nel tuo FullModel: loss_list[0]=semantic, loss_list[1]=boundary
#         if isinstance(loss_list, (list, tuple)) and len(loss_list) >= 2:
#             avg_sem_loss.update(loss_list[0].mean().item())
#             avg_bce_loss.update(loss_list[1].mean().item())

#         _ = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter + cur_iters)

#         # NEW: scheduler D
#         base_lr_D = getattr(config.TRAIN, 'LR_D', 1e-4)
#         power_D   = getattr(config.TRAIN, 'POWER_D', 1.0)
#         _ = adjust_learning_rate(optimizer_D, base_lr_D, num_iters, i_iter + cur_iters, power=power_D)

#         if i_iter % config.PRINT_FREQ == 0:
#             lr_g = [pg['lr'] for pg in optimizer.param_groups]
#             lr_d = [pg['lr'] for pg in optimizer_D.param_groups]
#             msg = (f"[ADV] Epoch: [{epoch}/{num_epoch}] Iter:[{i_iter}/{epoch_iters}], "
#                   f"LR_G:{lr_g} LR_D:{lr_d} "
#                   f"Loss(G): {ave_loss.average():.6f}, Adv(G): {(lambda_adv*loss_adv_G.item()):.6f}, "
#                   f"Loss(D): {loss_D.item():.6f}, Acc: {ave_acc.average():.6f}, "
#                   f"Sem: {avg_sem_loss.average():.6f}, Bnd: {avg_bce_loss.average():.6f}")
#             logging.info(msg)

#     writer.add_scalar('train_loss_adv', ave_loss.average(), global_steps)
#     writer_dict['train_global_steps'] = global_steps + 1


# =============================
# utils/adv_training.py  (loop di training avversario)
# =============================
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import AverageMeter, adjust_learning_rate
from configs import config
import time


# helper per scegliere la testa "main"
def _pick_main_from_seg_outputs(seg_outputs):
    # seg_outputs è la parte senza boundary: list/tuple di teste (es. [aux, main])
    if isinstance(seg_outputs, (list, tuple)):
        main_idx = getattr(config.TEST, 'OUTPUT_INDEX', len(seg_outputs) - 1)
        # fallback sicuro se l'indice non esiste
        if not (0 <= main_idx < len(seg_outputs)):
            main_idx = len(seg_outputs) - 1
        return seg_outputs[main_idx]
    return seg_outputs  # già tensore


def _freeze(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = not flag


def train_adv(config, epoch, num_epoch, 
          epoch_iters, base_lr, num_iters,
          trainloader, targetloader, 
          optimizer, optimizer_D1, 
          model, model_D1,
          writer_dict):
    
    model.train()
    model_D1.train()
    #model_D2.train()

    #bce_loss = torch.nn.MSELoss() if config.TRAIN.GAN == 'LS' else torch.nn.BCEWithLogitsLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()



    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_acc = AverageMeter()
    avg_sem_loss = AverageMeter()
    
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, (batch, target_batch) in enumerate(zip(trainloader, targetloader)):
        # 1. Training Generator
        optimizer.zero_grad()

        # Initialize loss values
        loss_seg_value1 = 0
        #loss_seg_value2 = 0
        loss_adv_target_value1 = 0
        #loss_adv_target_value2 = 0

        # Freeze discriminator
        for param in model_D1.parameters():
            param.requires_grad = False
        # for param in model_D2.parameters():
        #     param.requires_grad = False

        # Train with source domain
        images, labels, bd_gts, _, _ = batch
        images = images.cuda()
        labels = labels.long().cuda()
        bd_gts = bd_gts.float().cuda()
        h, w = labels.size(1), labels.size(2)

        losses_source, pred_source, acc_source, loss_list_source = model(images, labels, bd_gts)

        loss_seg_1 = losses_source.mean() 
        #loss_seg_2 = loss_list_source[2].mean()
        loss_seg = loss_seg_1 #+ config.TRAIN.LAMBDA_SEG2 * loss_seg_2
        loss_seg.backward()
        loss_seg_value1 += loss_seg_1.item()
        #loss_seg_value2 += loss_seg_2.item()

        # Train with target domain
        images_target, _, _, _, _ = target_batch
        images_target = images_target.cuda()

        _, pred_target, _, _ = model(images_target, labels, bd_gts)


        # Adversarial loss on target
        D_out_target_conv5 = model_D1(F.softmax(pred_target[1], dim=1))
        #D_out_target_conv4 = model_D2(F.softmax(pred_target[2], dim=1))

        loss_adv_target1 = config.LOSS.LAMBDA_ADV * bce_loss(D_out_target_conv5, 
                                                            torch.ones_like(D_out_target_conv5).cuda())
        # loss_adv_target2 = config.TRAIN.LAMBDA_ADV2 * bce_loss(D_out_target_conv4,
        #                                                     torch.ones_like(D_out_target_conv4).cuda())

        # Normalize and accumulate adversarial losses
        # loss_adv = (loss_adv_target1 + loss_adv_target2) 
        loss_adv = loss_adv_target1
        loss_adv.backward()
        loss_adv_target_value1 += loss_adv_target1.item() 
        #loss_adv_target_value2 += loss_adv_target2.item() 


        # 2. Training Discriminators
        for param in model_D1.parameters():
            param.requires_grad = True
        # for param in model_D2.parameters():
        #     param.requires_grad = True

        # Initialize loss values
        loss_D_value1 = 0
        # loss_D_value2 = 0

        # Train with source
        pred_source_conv5 = pred_source[1].detach()
        #pred_source_conv4 = pred_source[2].detach()

        D_out_source_conv5 = model_D1(F.softmax(pred_source_conv5, dim=1))
        #D_out_source_conv4 = model_D2(F.softmax(pred_source_conv4, dim=1))

        loss_D1_source = bce_loss(D_out_source_conv5, torch.ones_like(D_out_source_conv5).cuda())
        #loss_D2_source = bce_loss(D_out_source_conv4, torch.ones_like(D_out_source_conv4).cuda())

        # Train with target
        pred_target_conv5 = pred_target[1].detach()
        #pred_target_conv4 = pred_target[2].detach()

        D_out_target_conv5 = model_D1(F.softmax(pred_target_conv5, dim=1))
        #D_out_target_conv4 = model_D2(F.softmax(pred_target_conv4, dim=1))

        loss_D1_target = bce_loss(D_out_target_conv5, torch.zeros_like(D_out_target_conv5).cuda())
        #loss_D2_target = bce_loss(D_out_target_conv4, torch.zeros_like(D_out_target_conv4).cuda())

        # Combine and normalize losses
        optimizer_D1.zero_grad()
        loss_D1 = (loss_D1_source + loss_D1_target) / (2)
        loss_D1.backward()
        loss_D_value1 += loss_D1.item()
        optimizer_D1.step()

        # optimizer_D2.zero_grad()
        # loss_D2 = (loss_D2_source + loss_D2_target) / (2)
        # loss_D2.backward()
        # loss_D_value2 += loss_D2.item()
        # optimizer_D2.step()


        # Metrics update
        batch_time.update(time.time() - tic)
        tic = time.time()

        
        optimizer.step()
        ave_loss.update(loss_seg.item())
        ave_acc.update(acc_source.item())
        avg_sem_loss.update(loss_list_source[0].mean().item())
        
        lr = adjust_learning_rate(optimizer, base_lr, num_iters, i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}, Loss_D1: {:.6f}, Acc:{:.6f}, Semantic loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters,
                      batch_time.average(), [x['lr'] for x in optimizer.param_groups], 
                      ave_loss.average(), loss_D1.item(), #loss_D2.item(),
                      ave_acc.average(), avg_sem_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

