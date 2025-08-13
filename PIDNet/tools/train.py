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
from utils.function import train, validate
from utils.utils import create_logger, FullModel

import json
from utils.model_measures import count_params, measure_latency, count_flops
from utils.criterion import CrossEntropyLovasz, BondaryLoss

import math
from torch.amp import GradScaler as AmpGradScaler
scaler = AmpGradScaler('cuda')

import matplotlib.pyplot as plt

# >>> VISUALIZATION HELPERS (headless-safe)
import matplotlib
matplotlib.use("Agg")  # evita problemi su server senza display
import matplotlib.pyplot as plt


# dopo `import _init_paths` e gli altri import...
from utils.visualization import (
    save_samples,
    save_predictions_pidnet,
    LOVEDA_COLORMAP,
    # load_pidnet_from_checkpoint  # opzionale
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
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

    #Prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    trainloader = torch.utils.data.DataLoader(
          train_dataset,
          batch_size=batch_size,
          shuffle=config.TRAIN.SHUFFLE,
          num_workers=config.WORKERS,
          pin_memory=False,
          drop_last=True)

    #Visualization
    save_samples(
      train_dataset,
      n=6, cols=3, overlay=True, alpha=0.45,
      ignore_index=config.TRAIN.IGNORE_LABEL,  # nel tuo YAML è -1
      colormap=LOVEDA_COLORMAP,
      denorm=True,
      exp_name="pidnet_loveda_step_2b"         # cartella sotto PIDNet/output/<exp_name>/
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

    # COMPUTED WEIGHTS
    import os as _os, json as _json, numpy as _np

    def load_class_weights_any(path):
        ext = _os.path.splitext(path)[1].lower()
        if ext == ".json":
            obj = _json.load(open(path))
            arr = obj["values"] if isinstance(obj, dict) and "values" in obj else obj
            w = _np.array(arr, dtype=_np.float32).reshape(-1)
        elif ext == ".csv":
            w = _np.loadtxt(path, delimiter=",", dtype=_np.float32).reshape(-1)
        elif ext == ".txt":
            w = _np.loadtxt(path, dtype=_np.float32).reshape(-1)
        else:
            raise ValueError(f"Formato non supportato: {ext} (usa .json/.csv/.txt)")
        return torch.from_numpy(w)

    # === dopo i dataloader, prima di creare le loss:
    CLASS_WEIGHTS_PATH = "/content/drive/MyDrive/AML_Semantic_Segmentation/class_weights_invlog.json"  # o .csv/.txt
    train_dataset.class_weights = None
    if _os.path.exists(CLASS_WEIGHTS_PATH):
        w_t = load_class_weights_any(CLASS_WEIGHTS_PATH)
        # (opz.) w_t = torch.clamp(w_t, max=5.0) ; w_t = torch.pow(w_t, 0.5)
        train_dataset.class_weights = w_t.cuda()
        print(">> Using class weights:", train_dataset.class_weights.shape, train_dataset.class_weights.device)
    else:
        print(">> No class weights at", CLASS_WEIGHTS_PATH)

    # # criterion
    # if config.LOSS.USE_OHEM:
    #     sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
    #                                     thres=config.LOSS.OHEMTHRES,
    #                                     min_kept=config.LOSS.OHEMKEEP,
    #                                     weight=train_dataset.class_weights)
    # else:
    #     sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
    #                                 weight=train_dataset.class_weights)

    # === LOSS: CE (con o senza OHEM) + Lovasz (50/50)
    sem_criterion = CrossEntropyLovasz(
        num_classes=config.DATASET.NUM_CLASSES,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        balance_weights=config.LOSS.BALANCE_WEIGHTS,
        use_ohem=config.LOSS.USE_OHEM,         # se vuoi disattivare OHEM lascia False qui
        ohem_thres=config.LOSS.OHEMTHRES,
        ohem_keep=config.LOSS.OHEMKEEP,
        weight=getattr(train_dataset, 'class_weights', None),
        ce_weight=0.5,
        lovasz_weight=0.5,
        per_image=False,        # True se batch piccolo/variabile
        classes='present'       # considera solo classi presenti
    )


    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion, align_corners=config.MODEL.ALIGN_CORNERS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    opt_name = config.TRAIN.OPTIMIZER.lower()
    # param groups: niente weight decay su BN/bias
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    if opt_name in ['adamw', 'adam_w']:
        optimizer = torch.optim.AdamW(
            [{'params': decay, 'weight_decay': config.TRAIN.WD},
            {'params': no_decay, 'weight_decay': 0.0}],
            lr=config.TRAIN.LR,
            betas=getattr(config.TRAIN, 'BETAS', (0.9, 0.999))
        )
    elif opt_name in ['adam']:
        optimizer = torch.optim.Adam(
            [{'params': decay, 'weight_decay': config.TRAIN.WD},
            {'params': no_decay, 'weight_decay': 0.0}],
            lr=config.TRAIN.LR,
            betas=getattr(config.TRAIN, 'BETAS', (0.9, 0.999))
        )
    elif opt_name in ['sgd']:
        optimizer = torch.optim.SGD(
            [{'params': decay, 'weight_decay': config.TRAIN.WD},
            {'params': no_decay, 'weight_decay': 0.0}],
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            nesterov=config.TRAIN.NESTEROV
        )
    else:
        raise ValueError(f'Unsupported optimizer: {config.TRAIN.OPTIMIZER}')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch

    # === Scheduler: cosine per-iter con warmup (dopo real_end!)
    total_steps = int(real_end * epoch_iters)
    warmup_steps = int(0.03 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    #scaler = GradScaler(enabled=True)

    global_step = 0

    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch

    #Logging IoU per class
    class_names = ["Background","Building","Road","Water","Barren","Forest","Agriculture"]  # LoveDA

    for epoch in range(last_epoch, real_end):
        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train(
            config, epoch, config.TRAIN.END_EPOCH,
            epoch_iters, config.TRAIN.LR, total_steps,          # <--- usa total_steps
            trainloader, optimizer, model, writer_dict,
            scheduler=scheduler, scaler=scaler, clip_grad=5.0, amp=True
        )

        if flag_rm == 1 or (epoch % 5 == 0 and epoch < real_end - 100) or (epoch >= real_end - 100):
            valid_loss, mean_IoU, IoU_array = validate(config, 
                        testloader, model, writer_dict)
        if flag_rm == 1:
            flag_rm = 0

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
        }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best.pt'))
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)
        
                
        logging.info(msg)
        logging.info(IoU_array)

    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))
            
    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
