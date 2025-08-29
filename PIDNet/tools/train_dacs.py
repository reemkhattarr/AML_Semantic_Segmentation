import argparse
import os
import pprint
import logging
import timeit

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function_dacs import train_dacs, validate
from utils.utils import create_logger, FullModel

from datasets.dual_domain_loader import DualDomainLoader


def create_ema_model(student_model):
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    gpus = list(config.GPUS)
    ema_model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
    for ema_param, param in zip(ema_model.parameters(), student_model.module.model.parameters()):
        ema_param.data[:] = param.data[:].clone()
    ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus).cuda()
    for param in ema_model.parameters():
        param.detach_()
    return ema_model
        

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network with DACS')
    parser.add_argument('--cfg', help='experiment configure file name', default="configs/loveda/pidnet_loveda_4b.yaml", type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('--dacs-classmix-frac', type=float, default=0.5)
    parser.add_argument('--dacs-loss-weight', type=float, default=0.1)
    parser.add_argument('--dacs-pseudo-thr', type=float, default=0.968)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
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

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

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
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

    # Source: urban (labeled)
    src_dataset = eval('datasets.'+config.DATASET.DATASET)(
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
    src_loader = torch.utils.data.DataLoader(
        src_dataset, batch_size=batch_size, shuffle=True, num_workers=config.WORKERS, pin_memory=False, drop_last=True)

    # Target: rural (unlabeled)
    tgt_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TRAIN_SET_TARGET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        scale_factor=config.TRAIN.SCALE_FACTOR,
        #apply augmentations to target dataset?
        augmentation_type=None,
        #augmentation_type=config.TRAIN.get('AUGMENTATION_TYPE', None),
        aug_prob=config.TRAIN.get('AUG_PROB', 0.5)
    )
    tgt_loader = torch.utils.data.DataLoader(
        tgt_dataset, batch_size=batch_size, shuffle=True, num_workers=config.WORKERS, pin_memory=False, drop_last=True)

    dual_loader = DualDomainLoader(src_loader, tgt_loader)

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=config.DATASET.TEST_SET,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus), shuffle=False, num_workers=config.WORKERS, pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                         thres=config.LOSS.OHEMTHRES,
                                         min_kept=config.LOSS.OHEMKEEP,
                                         weight=src_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     weight=src_dataset.class_weights)
    bd_criterion = BondaryLoss()
    
    # Wrap student model in FullModel and DataParallel
    model = FullModel(model, sem_criterion, bd_criterion)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    
    
    teacher_model = create_ema_model(model)

    # optimizer
    params_dict = dict(model.named_parameters())
    params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]
    optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    
    
    epoch_iters = int(len(src_dataset) / batch_size)
    best_mIoU = 0
    last_epoch = 0

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH

    for epoch in range(last_epoch, end_epoch):
        train_dacs(
            config, epoch, end_epoch, epoch_iters, config.TRAIN.LR, end_epoch * epoch_iters,
            dual_loader, optimizer, model, teacher_model, writer_dict,
            pseudo_thr=args.dacs_pseudo_thr,
            dacs_loss_weight=args.dacs_loss_weight
        )


        # Validate student model
        valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)
        

        logger.info('=> saving checkpoint to {}'.format(final_output_dir + 'checkpoint.pth.tar'))
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

        msg = 'Loss: {:.3f}, MeanIU: {:4.4f}, Best_mIoU: {:4.4f}'.format(
                valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info('IoU: {}'.format(IoU_array))
    
    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))
    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
