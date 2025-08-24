# =============================
# train_adv.py  (entrypoint separato)
# =============================
# Avvia il training con adattamento avversario in "output space" SENZA toccare il train standard.
# Usa: python train_adv.py --cfg configs/loveda/pidnet_loveda_2b.yaml

import argparse
import os
import pprint
import logging
import timeit

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths  # come nel tuo train.py
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import validate  # riuso la tua validate
from utils.utils import create_logger, FullModel

from models.discriminator import FCDiscriminator
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network (ADV)')
    parser.add_argument('--cfg', default="configs/loveda/pidnet_loveda_2b.yaml", type=str,
                        help='config file')
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train_adv')
    logger.info(pprint.pformat(args))
    logger.info(config)

    from tensorboardX import SummaryWriter
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0

    # -------------------------
    # Modello di segmentazione
    # -------------------------
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    seg = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)

    # Datasets (SOURCE = train standard)
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    crop_size  = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR,
        augmentation_type=config.TRAIN.get('AUGMENTATION_TYPE', None),
        aug_prob=config.TRAIN.get('AUG_PROB', 0.5)
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)

    # TARGET = dataset del dominio target (se non specificato, fallback al train_set)
    target_list = getattr(config.DATASET, 'TRAIN_SET_TARGET', None) or config.DATASET.TRAIN_SET
    target_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=target_list,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR,
        augmentation_type=config.TRAIN.get('AUGMENTATION_TYPE', None),
        aug_prob=config.TRAIN.get('AUG_PROB', 0.5)
    )

    targetloader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)

    # Val/TEST loader
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # Criteri "classici"
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                         thres=config.LOSS.OHEMTHRES,
                                         min_kept=config.LOSS.OHEMKEEP,
                                         weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     weight=train_dataset.class_weights)
    bd_criterion = BondaryLoss()

    model = FullModel(seg, sem_criterion, bd_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Optimizer segmenter (SGD come nel tuo codice)
    assert config.TRAIN.OPTIMIZER == 'sgd', 'Only Support SGD optimizer here.'
    params_dict = dict(model.named_parameters())
    params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]
    optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV)

    # -------------------------
    # Discriminatore + optimizer
    # -------------------------
    D = FCDiscriminator(num_classes=config.DATASET.NUM_CLASSES)
    D = nn.DataParallel(D, device_ids=gpus).cuda()

    # usa il LR_D dal config
    lr_d = getattr(config.TRAIN, 'LR_D', 1e-4)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.9, 0.99))
    bce_adv = nn.BCEWithLogitsLoss().cuda()
    lambda_adv = getattr(config.LOSS, 'LAMBDA_ADV', 0.01)

    # Resume (facoltativo): separiamo i checkpoint ADV per non intaccare il training classico
    best_mIoU = 0
    last_epoch = 0
    ckpt_path = os.path.join(final_output_dir, 'checkpoint_adv.pth.tar')
    if config.TRAIN.RESUME and os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location={'cuda:0': 'cpu'}, weights_only=False)
        best_mIoU = checkpoint['best_mIoU']
        last_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['seg_state'])
        D.module.load_state_dict(checkpoint['disc_state'])
        optimizer.load_state_dict(checkpoint['optim_seg'])
        optimizer_D.load_state_dict(checkpoint['optim_D'])
        logger.info(f"=> loaded ADV checkpoint (epoch {checkpoint['epoch']})")

    # Loop
    from utils.utils import adjust_learning_rate
    from utils.adv_training import train_adv

    epoch_iters = int(len(train_dataset) / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = end_epoch * epoch_iters

    # early stopping
    patience = 12
    epochs_no_improvement = 0

    start = timeit.default_timer()
    for epoch in range(last_epoch, end_epoch):
        if trainloader.sampler is not None and hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)
        if targetloader.sampler is not None and hasattr(targetloader.sampler, 'set_epoch'):
            targetloader.sampler.set_epoch(epoch)

        # train_adv(config, epoch, end_epoch, epoch_iters, config.TRAIN.LR, num_iters,
        #           trainloader, targetloader, optimizer, optimizer_D,
        #           model, D, bce_adv, lambda_adv, writer_dict)

        train_adv(config, epoch, end_epoch, epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, targetloader, optimizer, optimizer_D,
                  model, D, writer_dict)        

        valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)

        # Save ADV checkpoint
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'seg_state': model.module.state_dict(),
            'disc_state': D.module.state_dict(),
            'optim_seg': optimizer.state_dict(),
            'optim_D': optimizer_D.state_dict(),
        }, ckpt_path)

        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'best_adv.pt'))
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            logger.info(f'No improvement in mIoU for {epochs_no_improvement} epoch(s)')

        if epochs_no_improvement > patience:
            logger.info(f'Early stopping after {epochs_no_improvement} epochs without improvement.')
            break

        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
            valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info(IoU_array)

    # salvataggio finale + restore best
    torch.save(model.module.state_dict(), os.path.join(final_output_dir, 'final_state_adv.pt'))
    best_model_path = os.path.join(final_output_dir, 'best_adv.pt')
    if os.path.exists(best_model_path):
        model.module.load_state_dict(torch.load(best_model_path))
        logger.info("Best ADV model weights restored after early stopping.")

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done (ADV)')


if __name__ == '__main__':
    main()