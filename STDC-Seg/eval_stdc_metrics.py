#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ========= Utils: FLOPs / Params =========
def get_flops_params(model, input_shape):
    # Prefer thop, fallback a ptflops
    flops = None
    params = sum(p.numel() for p in model.parameters())
    try:
        from thop import profile
        dummy = torch.randn(*input_shape, device=next(model.parameters()).device)
        model.eval()
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        try:
            from ptflops import get_model_complexity_info
            # ptflops vuole (C,H,W)
            c, h, w = input_shape[1], input_shape[2], input_shape[3]
            macs, params_pf = get_model_complexity_info(
                model, (c, h, w), as_strings=False, print_per_layer_stat=False, verbose=False
            )
            # ptflops ritorna MACs: FLOPs ≈ 2 * MACs in molte convenzioni, ma qui usiamo MACs come "FLOPs eq."
            # Per consistenza col resto, consideriamo macs come operazioni (MAC).
            flops = macs
            params = int(params_pf)
        except Exception:
            pass
    return flops, params

# ========= Utils: Latency =========
def measure_latency(model, input_shape, warmup=20, iters=100):
    device = next(model.parameters()).device
    x = torch.randn(*input_shape, device=device)
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    avg_ms = (t1 - t0) * 1000.0 / iters
    return avg_ms

# ========= mIoU: Confusion Matrix =========
class ConfusionMatrix:
    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, preds, targets):
        """
        preds: (N,H,W) int64
        targets: (N,H,W) int64
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]
        # Filtra eventuali label fuori range
        valid = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]
        k = self.num_classes
        idx = targets * k + preds
        bincount = torch.bincount(idx, minlength=k * k)
        self.mat += bincount.view(k, k).cpu()

    def compute_iou(self):
        h = self.mat
        diag = torch.diag(h).to(torch.float64)
        denom = (h.sum(1) + h.sum(0) - diag).clamp_min(1).to(torch.float64)
        iou = diag / denom
        return iou

    def miou(self):
        iou = self.compute_iou()
        return float(iou.mean().item()), iou.tolist()

# ========= Data: LoveDA (esempio) =========
def build_dataset(name, root, split, img_size=None, ignore_index=255):
    """
    Adatta questa funzione al tuo dataset.
    Per LoveDA, si assume una classe dataset compatibile che ritorna:
      dict/image tensor (C,H,W) normalizzato e target long (H,W)
    """
    if name.lower() == "loveda":
        # Esempio: datasets.loveda.LoveDA(split='val', ...)
        from datasets.loveda import LoveDA  # <-- adattala al tuo path
        ds = LoveDA(root=root, split=split, img_size=img_size, ignore_index=ignore_index)
        return ds
    else:
        raise ValueError(f"Dataset '{name}' non supportato qui. Adatta build_dataset().")

# ========= Model loading =========
def load_model_from_builder(builder_path, num_classes, checkpoint=None, device="cuda"):
    """
    builder_path: stringa tipo 'your_pkg.models.stdc:build_stdc2_seg'
    dove la funzione deve restituire un nn.Module con num_classes già impostate,
    oppure accettare num_classes come argomento.
    """
    module_name, func_name = builder_path.split(":")
    mod = importlib.import_module(module_name)
    builder = getattr(mod, func_name)

    try:
        model = builder(num_classes=num_classes)
    except TypeError:
        # Nel caso la tua builder non preveda num_classes
        model = builder()

    model.to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        # Prova a gestire vari formati comuni
        state = ckpt.get("state_dict") or ckpt.get("model_state") or ckpt
        # Rimuovi eventuali prefissi 'module.'
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {len(missing)} (most common with strict=False)")
        if unexpected:
            print(f"[WARN] Unexpected keys: {len(unexpected)}")
    return model

# ========= Inference helper =========
@torch.no_grad()
def predict_logits(model, imgs, amp=False):
    if amp:
        with torch.autocast(device_type=next(model.parameters()).device.type, dtype=torch.float16):
            out = model(imgs)
    else:
        out = model(imgs)
    # Se il modello ritorna tuple/dict, prova a prendere la testa principale
    if isinstance(out, (list, tuple)):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("out") or out.get("logits") or list(out.values())[0]
    return out

# ========= Eval mIoU =========
def evaluate_miou(model, loader, num_classes, ignore_index=255, amp=False):
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            # Supporta dataset che ritornano tuple o dict
            if isinstance(batch, dict):
                imgs = batch.get("image") or batch.get("img") or batch.get("images")
                targets = batch.get("mask") or batch.get("label") or batch.get("target")
            else:
                imgs, targets = batch

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = predict_logits(model, imgs, amp=amp)  # (N,C,H',W')
            # Upsample a (H,W) del target se necessario
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1).long()
            cm.update(preds.cpu(), targets.cpu())

    miou, per_class = cm.miou()
    return miou, per_class

def main():
    parser = argparse.ArgumentParser("Eval STDC: mIoU, Latency, FLOPs, Params")
    # Modello
    parser.add_argument("--model_builder", type=str, required=True,
                        help="Es: your_pkg.models.stdc:build_stdc2_seg")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path .pth del modello")
    parser.add_argument("--num_classes", type=int, required=True)
    # Dataset / mIoU
    parser.add_argument("--dataset", type=str, default="loveda", help="Nome dataset (es: loveda)")
    parser.add_argument("--data_root", type=str, default=None, help="Root del dataset")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--img_size", type=int, nargs=2, default=None, help="(H W) resize opzionale per dataset")
    parser.add_argument("--amp", type=lambda s: s.lower() in ["1","true","t","yes","y"], default=False)
    # FLOPs/Params/Latency
    parser.add_argument("--input_size", type=int, nargs=2, default=[512, 512],
                        help="(H W) input per FLOPs/Latency")
    parser.add_argument("--lat_warmup", type=int, default=20)
    parser.add_argument("--lat_iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model_from_builder(args.model_builder, args.num_classes, args.checkpoint, device=device)

    # ---- Params / FLOPs ----
    in_shape = (1, 3, args.input_size[0], args.input_size[1])
    flops, params = get_flops_params(model, in_shape)
    if flops is not None:
        print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
    else:
        print("FLOPs: N/A (installa 'thop' o 'ptflops')")
    print(f"Params: {params/1e6:.2f} M")

    # ---- Latency ----
    try:
        latency_ms = measure_latency(model, in_shape, warmup=args.lat_warmup, iters=args.lat_iters)
        print(f"Latency (1 image {args.input_size[0]}x{args.input_size[1]}): {latency_ms:.2f} ms")
    except Exception as e:
        print(f"Latency: errore durante la misura: {e}")

    # ---- mIoU (opzionale se data_root fornita) ----
    if args.data_root is not None:
        ds = build_dataset(args.dataset, args.data_root, args.split,
                           img_size=tuple(args.img_size) if args.img_size else None,
                           ignore_index=args.ignore_index)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        miou, per_class = evaluate_miou(model, loader, args.num_classes, ignore_index=args.ignore_index, amp=args.amp)
        print(f"mIoU: {miou*100:.2f}%")
        print("Class IoU:", ", ".join(f"{v*100:.2f}%" for v in per_class))
    else:
        print("mIoU: skipped (nessun --data_root)")

if __name__ == "__main__":
    main()
