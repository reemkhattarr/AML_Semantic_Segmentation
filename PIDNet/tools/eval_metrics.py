# ------------------------------------------------------------------------------
# Eval metrics standalone for PIDNet (Params, Latency, FLOPs)
# Basato sulla tua sezione finale di train.py, ma indipendente.
# ------------------------------------------------------------------------------

import argparse
import os
import json
import torch

# imposta i path locali del repo come nel train
import _init_paths  # noqa: F401
import models
from configs import config
from configs import update_config
from utils.utils import create_logger
from utils.model_measures import count_params, measure_latency, count_flops

import torch.nn as nn
import datasets
from tensorboardX import SummaryWriter
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import validate
from utils.utils import FullModel


def parse_args():
    parser = argparse.ArgumentParser(description="Calcola metriche modello (post-training)")
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
        help="Path al file di config (.yaml) usato in training",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path ai pesi del modello (best.pt / final_state.pt / checkpoint.pth.tar). "
             "Se omesso, cerca in automatico nella output dir del cfg.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo per la misurazione (default: auto)",
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="Iterazioni per la misurazione della latency"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Iterazioni di warmup per la latency"
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Batch usato per l'input fittizio (default 1)"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Forza AMP per la latency (altrimenti auto se CUDA)"
    )
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Override da riga di comando, es. KEY VALUE ..."
    )
    parser.add_argument("--no-miou", dest="miou", action="store_false",
                    help="Non calcolare la mIoU"
    )
    parser.set_defaults(miou=True)

    return parser.parse_args()


def resolve_device(arg_device: str) -> str:
    if arg_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg_device


def load_state_dict_flexible(weight_path: str):
    """Carica in modo flessibile state_dict da .pt/.tar, gestendo DataParallel/FullModel."""
    raw = torch.load(weight_path, map_location="cpu")

    # se è un checkpoint {"state_dict": ..., "epoch": ...}
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    else:
        sd = raw

    # togli eventuale prefisso 'module.' (DataParallel)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # se proviene da FullModel, tieni solo 'model.*' e rimuovi il prefisso
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}

    return sd


def main():
    args = parse_args()
    if args.opts is None:
        args.opts = []
    update_config(config, args)


    # crea logger e risolve la cartella di output coerente con il training
    # (usiamo una fase chiamata 'eval' per non sovrascrivere i log di train)
    logger, final_output_dir, tb_dir = create_logger(config, args.cfg, "eval")

    # individua i pesi
    weight_path = args.weights
    if not weight_path:
        # prova best.pt poi final_state.pt poi checkpoint.pth.tar nella output dir
        candidates = [
            os.path.join(final_output_dir, "best.pt"),
            os.path.join(final_output_dir, "final_state.pt"),
            os.path.join(final_output_dir, "checkpoint.pth.tar"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                weight_path = c
                break

    if not weight_path or not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Non trovo i pesi. Passa --weights o assicurati che "
            f"'best.pt' / 'final_state.pt' / 'checkpoint.pth.tar' esistano in {final_output_dir}"
        )

    logger.info(f"=> Uso config: {args.cfg}")
    logger.info(f"=> Carico pesi: {weight_path}")

    # ricrea il modello "puro" (senza FullModel/DataParallel) e carica i pesi
    imgnet = "imagenet" in config.MODEL.PRETRAINED
    eval_model = models.pidnet.get_seg_model(config, imgnet_pretrained=False if imgnet else False)

    sd = load_state_dict_flexible(weight_path)
    eval_model.load_state_dict(sd, strict=True)
    eval_model.eval()

    # input size: come nel test (H, W) dal config
    # in train.py: test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    H, W = config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0]
    input_size = (args.batch, 3, H, W)
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    use_amp = args.amp or (device == "cuda")


    # ---- Params
        # ---- (opzionale) mIoU su test set
    miou = None
    class_iou = None
    if args.miou:
        test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
        test_dataset = eval('datasets.' + config.DATASET.DATASET)(
            root=config.DATASET.ROOT,
            list_path=config.DATASET.TEST_SET,
            num_classes=config.DATASET.NUM_CLASSES,
            multi_scale=False,
            flip=False,
            ignore_label=config.TRAIN.IGNORE_LABEL,
            base_size=config.TEST.BASE_SIZE,
            crop_size=test_size
        )

        gpus = list(config.GPUS)
        cuda_ok = torch.cuda.is_available()
        # Se il numero di GPU nel cfg non coincide con quelle disponibili, adattiamo
        device_count = torch.cuda.device_count() if cuda_ok else 0
        if cuda_ok and device_count != len(gpus):
            logger.warning(f"GPU config vs disponibili: cfg={len(gpus)}, reali={device_count}. Adatto i device_ids.")
            gpus = list(range(device_count))

        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.TEST.BATCH_SIZE_PER_GPU * (len(gpus) if cuda_ok else 1),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=False
        )

        # criteri come nel train.py
        if config.LOSS.USE_OHEM:
            sem_criterion = OhemCrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                thres=config.LOSS.OHEMTHRES,
                min_kept=config.LOSS.OHEMKEEP,
                weight=getattr(test_dataset, "class_weights", None)
            )
        else:
            sem_criterion = CrossEntropy(
                ignore_label=config.TRAIN.IGNORE_LABEL,
                weight=getattr(test_dataset, "class_weights", None)
            )
        bd_criterion = BondaryLoss()

        # wrapper per riusare validate()
        eval_wrapper = FullModel(eval_model, sem_criterion, bd_criterion)
        if cuda_ok and len(gpus) > 0:
            eval_wrapper = nn.DataParallel(eval_wrapper, device_ids=gpus).cuda()

        writer_dict = {
            'writer': SummaryWriter(tb_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0
        }
        
        with torch.no_grad():
            valid_loss, mean_IoU, IoU_array = validate(config, testloader, eval_wrapper, writer_dict)

        writer_dict['writer'].close()

        miou = float(mean_IoU) * 100.0
        class_iou = [float(x) * 100.0 for x in IoU_array]

    # ---- Latency (AMP su GPU se disponibile)
    H, W = config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0]
    input_size = (args.batch, 3, H, W)
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    use_amp = args.amp or (device == "cuda")

    lat = measure_latency(
        eval_model,
        input_size=input_size,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        amp=use_amp,
    )

    # ---- FLOPs
    gflops, pretty_flops = count_flops(eval_model, input_size=input_size, device=device)

    # ---- Params
    n_params, pretty_params = count_params(eval_model, trainable_only=False)

    # === LOG in ordine: mIoU -> Latency -> FLOPs -> Params
    logger.info("\n=== Model Metrics (eval standalone) ===")
    if miou is not None:
        logger.info(f"mIoU:       {miou:.2f}%")
    logger.info(
        f"Latency:    avg {lat['avg_ms']:.2f} ms | p50 {lat['p50_ms']:.2f} | p95 {lat['p95_ms']:.2f} "
        f"| Throughput {lat['throughput_fps']:.2f} FPS"
    )
    logger.info(f"FLOPs:      {pretty_flops}")
    logger.info(f"Params:     {pretty_params} ({int(n_params)})")

    # === JSON con stesso ordine logico (Python 3.7+ preserva l'ordine di inserimento)
    results = {
        "cfg": os.path.basename(args.cfg),
        "weights": os.path.basename(weight_path),
        "input_size": list(input_size),
        "device": device,
        "amp": bool(use_amp),

        # misure nell'ordine richiesto
        "miou_pct": miou,                    # può essere None se --miou non passato
        "latency_ms": {
            "avg": float(lat["avg_ms"]),
            "p50": float(lat["p50_ms"]),
            "p95": float(lat["p95_ms"]),
        },
        "throughput_fps": float(lat["throughput_fps"]),
        "flops_g": float(gflops),
        "flops_pretty": pretty_flops,
        "params": {"count": int(n_params), "pretty": pretty_params},
        "class_iou_pct": class_iou,          # opzionale: elenco IoU di classe in %
    }

    out_json = os.path.join(final_output_dir, "model_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"=> Metriche salvate in: {out_json}")


if __name__ == "__main__":
    main()
