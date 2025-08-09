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
    # prepare data
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
                                        weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)

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
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict)

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
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))
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

    # ---------------------------
    # MISURE: params, latency, FLOPs
    # ---------------------------
    try:
        # ricreo il modello "puro" (senza wrapper FullModel / DataParallel)
        eval_model = models.pidnet.get_seg_model(config, imgnet_pretrained=False)

        # carico i pesi migliori se disponibili, altrimenti gli ultimi
        weight_path = os.path.join(final_output_dir, 'best.pt')
        if not os.path.isfile(weight_path):
            weight_path = os.path.join(final_output_dir, 'final_state.pt')

        raw = torch.load(weight_path, map_location='cpu')

        # 1) se è un checkpoint con 'state_dict', prendilo
        sd = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw

        # 2) rimuovi eventuale prefisso 'module.' (DataParallel)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}

        # 3) se è lo state_dict di FullModel, tieni SOLO le chiavi 'model.*' e togli 'model.'
        if any(k.startswith('model.') for k in sd.keys()):
            sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}

        # 4) carica nel modello puro
        eval_model.load_state_dict(sd, strict=True)


        device = "cuda" if torch.cuda.is_available() else "cpu"
        # usa la test size dal config: test_size = (H, W) creato sopra
        B, C = 1, 3
        H, W = test_size  # definita prima nel tuo script
        input_size = (B, C, H, W)

        # conta parametri (tutti, anche non addestrabili)
        n_params, pretty_params = count_params(eval_model, trainable_only=False)

        # latency (usa AMP su GPU se disponibile)
        lat = measure_latency(
            eval_model,
            input_size=input_size,
            device=device,
            warmup=10,
            iters=50,
            amp=(device == "cuda")
        )

        # FLOPs (per forward single-batch)
        gflops, pretty_flops = count_flops(eval_model, input_size=input_size, device=device)

        results = {
            "weights": os.path.basename(weight_path),
            "input_size": list(input_size),
            "device": device,
            "params": {"count": int(n_params), "pretty": pretty_params},
            "latency_ms": {"avg": lat["avg_ms"], "p50": lat["p50_ms"], "p95": lat["p95_ms"]},
            "throughput_fps": lat["throughput_fps"],
            "flops_g": gflops,
            "flops_pretty": pretty_flops,
            "amp": (device == "cuda"),
        }

        # stampa carina a log
        logger.info("\n=== Model Metrics (post-training) ===")
        logger.info(f"weights:    {results['weights']}")
        logger.info(f"input:      {results['input_size']}")
        logger.info(f"params:     {results['params']['pretty']} ({results['params']['count']})")
        logger.info(f"latency:    avg {results['latency_ms']['avg']:.2f} ms | "
                    f"p50 {results['latency_ms']['p50']:.2f} | p95 {results['latency_ms']['p95']:.2f}")
        logger.info(f"throughput: {results['throughput_fps']:.2f} FPS")
        logger.info(f"FLOPs:      {results['flops_pretty']}")

        # salvataggio JSON in output dir
        with open(os.path.join(final_output_dir, "model_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metriche salvate in {os.path.join(final_output_dir, 'model_metrics.json')}")

    except Exception as e:
        logger.error(f"[MISURE] Errore durante la misurazione: {e}")


    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    main()
