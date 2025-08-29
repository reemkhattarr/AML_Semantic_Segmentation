# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function_adv_new import train_adversarial, validate
from utils.utils import create_logger, FullModel
from models.discriminator import FCDiscriminator


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/loveda/pidnet_loveda_4a.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

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

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)

    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    
    # Prepare source dataset and dataloader (urban, labeled)
    train_dataset_source = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET,  # urban list
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
    trainloader_source = torch.utils.data.DataLoader(
        train_dataset_source,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)
        
    # Prepare target dataset and dataloader (rural, unlabeled)
    train_dataset_target = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET_TARGET,  # rural list
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR,
        augmentation_type=None,
        aug_prob=0.0
    )
    trainloader_target = torch.utils.data.DataLoader(
        train_dataset_target,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)
        
    # Initialize discriminator and optimizer
    model_domain = FCDiscriminator(num_classes=config.DATASET.NUM_CLASSES)
    model_domain = model_domain.cuda()
    domain_optimizer = torch.optim.Adam(
        model_domain.parameters(),
        lr=config.TRAIN.LR_D,
        weight_decay=0
    )


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

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset_source.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset_source.class_weights)

    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset_source.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'}, weights_only=False)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    # for early stopping and saving best model
    patience = 12
    epochs_no_improvement = 0
    
    for epoch in range(last_epoch, real_end):
            
        current_trainloader_source = trainloader_source
        if current_trainloader_source.sampler is not None and hasattr(current_trainloader_source.sampler, 'set_epoch'):
            current_trainloader_source.sampler.set_epoch(epoch)
            
        current_trainloader_target = trainloader_target
        if current_trainloader_target.sampler is not None and hasattr(current_trainloader_target.sampler, 'set_epoch'):
            current_trainloader_target.sampler.set_epoch(epoch)


        # Train adversarially with both loaders and optimizers
        train_adversarial(config, epoch, config.TRAIN.END_EPOCH, epoch_iters,
                          config.TRAIN.LR, num_iters,
                          trainloader_source, trainloader_target,
                          optimizer, domain_optimizer,
                          model, model_domain,
                          writer_dict)

        # Validation as usual
        valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)
        
        if flag_rm == 1:
            flag_rm = 0

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best.pt'))
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


    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))
    
    # Restore best model weights (optional)
    best_model_path = os.path.join(final_output_dir, 'best.pt')
    if os.path.exists(best_model_path):
        model.module.load_state_dict(torch.load(best_model_path))
        logger.info("Best model weights restored after early stopping.")

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')


if __name__ == '__main__':
    main()
