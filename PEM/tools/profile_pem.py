# tools/profile_pem.py
import os, sys, time, argparse
import torch

# --- rende importabile il pacchetto "pem" quando lanci da repo root ---
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# registra backbone / heads custom
import pem.modeling  # noqa

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from fvcore.nn import FlopCountAnalysis

from detectron2.config import get_cfg, CfgNode as CN

def build_cfg(args):
    cfg = get_cfg()

    # consenti chiavi nuove durante il merge (evita KeyError su campi di versioni diverse)
    cfg.set_new_allowed(True)

    # registra le estensioni del repo
    try:
        import pem.modeling  # registra backbone/testa PEM
        import pem.config as pem_cfg
        if hasattr(pem_cfg, "add_maskformer2_config"):
            pem_cfg.add_maskformer2_config(cfg)
        elif hasattr(pem_cfg, "add_pem_config"):
            pem_cfg.add_pem_config(cfg)
    except Exception as e:
        print(f"[profile] estensioni non caricate: {e}")

    # Fallback: assicurati che il nodo RESNETS esista e abbia i campi base
    cfg.defrost()
    if not hasattr(cfg.MODEL, "RESNETS"):
        cfg.MODEL.RESNETS = CN()
    if "STEM_TYPE" not in cfg.MODEL.RESNETS:
        cfg.MODEL.RESNETS.STEM_TYPE = "basic"
    if "STEM_OUT_CHANNELS" not in cfg.MODEL.RESNETS:
        cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

    # ora puoi fare il merge dello YAML senza errori
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    # device / weights
    cfg.MODEL.DEVICE = args.device
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    # fallback mask format se manca
    if not hasattr(cfg.INPUT, "MASK_FORMAT"):
        cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.freeze()
    return cfg

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--input-size", nargs=2, type=int, default=[1024, 1024])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cfg = build_cfg(args)

    # build & load
    model = build_model(cfg)
    model.eval()
    model.to(args.device)
    if args.weights:
        DetectionCheckpointer(model).load(args.weights)

    H, W = args.input_size
    # Detectron2 si aspetta una lista di dict con "image", "height", "width"
    dummy = torch.randn(1, 3, H, W, device=args.device)
    inputs = [{"image": dummy[0], "height": H, "width": W}]

    # ----- Params -----
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params:.2f} M")

    # ----- FLOPs (può sottostimare se ci sono op non supportati, es. DeformConv) -----
    try:
        flops = FlopCountAnalysis(model, inputs).total() / 1e9
        print(f"FLOPs:  {flops:.2f} GFLOPs @ {W}×{H}")
    except Exception as e:
        print(f"FLOPs:  N/A (errore: {e})")

    # ----- Latency (ms, batch=1) -----
    # warmup
    for _ in range(args.warmup):
        _ = model(inputs)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args.runs):
        _ = model(inputs)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    dt = (time.time() - t0) / args.runs * 1000.0
    print(f"Latency (1 image): {dt:.2f} ms")

    # Nota: per mIoU usa la pipeline di valutazione:
    #   python train_net.py --config-file ... --eval-only MODEL.WEIGHTS <ckpt>
    # Questo profiler non scorre il dataset.

if __name__ == "__main__":
    main()
