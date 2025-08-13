# --- PIDNet Visualizer (SAVE TO PIDNet/output) --------------------------------
import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List

# Colormap LoveDA (7 classi) – personalizzabile
LOVEDA_COLORMAP = np.array([
    [255, 255, 255],  # 0: Background
    [255,   0,   0],  # 1: Building
    [  0, 255,   0],  # 2: Road
    [  0,   0, 255],  # 3: Water
    [255, 255,   0],  # 4: Barren
    [  0, 255, 255],  # 5: Forest
    [255,   0, 255],  # 6: Agriculture
], dtype=np.uint8)

# Statistiche ImageNet per denormalizzare (se usate)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def _guess_pidnet_root_from_here() -> str:
    """Prova a risalire fino alla cartella PIDNet per poi usare PIDNet/output/."""
    # 1) se esegui da file in repo
    try:
        here = os.path.abspath(os.path.dirname(__file__))
    except NameError:
        here = os.getcwd()
    path = here
    while True:
        base = os.path.basename(path)
        if base == "PIDNet" and os.path.isdir(os.path.join(path, "output")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            # fallback: se non trovo PIDNet, uso cwd
            return here
        path = parent

def _default_output_root() -> str:
    pidnet_root = _guess_pidnet_root_from_here()
    out = os.path.join(pidnet_root, "output")
    os.makedirs(out, exist_ok=True)
    return out

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# -----------------------------------------------------------------------------
# Utils immagine
# -----------------------------------------------------------------------------
def decode_segmap(
    mask: np.ndarray,
    colormap: Optional[np.ndarray] = None,
    ignore_index: Optional[int] = None
) -> np.ndarray:
    """Converte una mask (H,W) o (N,H,W) di class index in RGB colorato."""
    if colormap is None:
        colormap = LOVEDA_COLORMAP
    single = (mask.ndim == 2)
    if single:
        mask = mask[None, ...]
    N, H, W = mask.shape
    out = np.zeros((N, H, W, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(colormap):
        out[mask == cls_idx] = color
    if ignore_index is not None:
        out[mask == ignore_index] = (0, 0, 0)
    return out[0] if single else out

def denormalize_img(img_chw: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """
    img_chw: (3,H,W) -> (H,W,3) in [0,1]
    """
    img_hwc = np.transpose(img_chw, (1,2,0))
    img_hwc = img_hwc * std + mean
    return np.clip(img_hwc, 0, 1)

def _smart_unpack_item(sample):
    """
    Supporta:
      - (image, mask)
      - (image, mask, edge, size, name)
    """
    if isinstance(sample, (list, tuple)):
        if len(sample) >= 2:
            image = sample[0]
            mask  = sample[1]
            name  = sample[4] if len(sample) >= 5 else None
            return image, mask, name
    raise ValueError("Formato campione non riconosciuto. Attesi (img, mask) o (img, mask, edge, size, name).")

def _smart_unpack_batch(batch):
    """
    Supporta batch:
      - (images, masks)
      - (images, masks, edges, sizes, names)
    """
    if isinstance(batch, (list, tuple)):
        images = batch[0]
        masks  = batch[1]
        names  = batch[4] if len(batch) >= 5 else None
        return images, masks, names
    raise ValueError("Formato batch non riconosciuto.")


# -----------------------------------------------------------------------------
# SALVA CAMPIONI (dataset) – senza modello
# -----------------------------------------------------------------------------
def save_samples(
    dataset,
    n: int = 6,
    cols: int = 3,
    overlay: bool = True,
    alpha: float = 0.45,
    ignore_index: int = 255,
    colormap: Optional[np.ndarray] = None,
    denorm: bool = True,
    save_dir: Optional[str] = None,
    exp_name: str = "colab_viz"
) -> List[str]:
    """
    Salva n campioni del dataset in: PIDNet/output/<exp_name>/samples_<timestamp>/
    Ritorna i path dei file salvati.
    """
    if save_dir is None:
        out_root = _default_output_root()
        save_dir = _ensure_dir(os.path.join(out_root, exp_name, f"samples_{_timestamp()}"))
    else:
        save_dir = _ensure_dir(save_dir)

    saved_paths = []
    rows = int(np.ceil(n / cols))

    # Salva anche una griglia complessiva
    fig_grid = plt.figure(figsize=(cols * 4, rows * 4))
    for i in range(min(n, len(dataset))):
        img_t, gt_t, name = _smart_unpack_item(dataset[i])
        img_np = img_t.detach().cpu().numpy()
        gt_np  = gt_t.detach().cpu().numpy() if hasattr(gt_t, "detach") else np.array(gt_t)

        img_vis = denormalize_img(img_np) if denorm else np.transpose(img_np, (1,2,0))
        gt_vis  = decode_segmap(gt_np, colormap=colormap, ignore_index=ignore_index)

        # Salva per-sample (side-by-side Input / GT overlay)
        fig, ax = plt.subplots(1, 2 if overlay else 1, figsize=(8 if overlay else 4, 4))
        if overlay:
            ax0, ax1 = ax
            ax0.imshow(img_vis); ax0.set_title('Input'); ax0.axis('off')
            ax1.imshow(img_vis); ax1.imshow(gt_vis, alpha=alpha); ax1.set_title('GT overlay'); ax1.axis('off')
        else:
            ax.imshow(img_vis); ax.set_title('Input'); ax.axis('off')

        fname = f"{i:03d}_{str(name) if name is not None else 'sample'}.png"
        fpath = os.path.join(save_dir, fname)
        plt.tight_layout()
        fig.savefig(fpath, dpi=120)
        plt.close(fig)
        saved_paths.append(fpath)

        # Aggiungi al mosaico complessivo
        axm = fig_grid.add_subplot(rows, cols, i + 1)
        axm.imshow(img_vis)
        if overlay:
            axm.imshow(gt_vis, alpha=alpha)
        axm.set_title(str(name) if name is not None else f"sample {i}")
        axm.axis('off')

    fig_grid.tight_layout()
    grid_path = os.path.join(save_dir, "_grid.png")
    fig_grid.savefig(grid_path, dpi=120)
    plt.close(fig_grid)
    saved_paths.append(grid_path)

    print(f"[save_samples] Saved {len(saved_paths)-1} samples + grid to: {save_dir}")
    return saved_paths


# -----------------------------------------------------------------------------
# SALVA PREDIZIONI PIDNet – con modello
# -----------------------------------------------------------------------------
@torch.no_grad()
def save_predictions_pidnet(
    model: torch.nn.Module,
    data: Union[torch.utils.data.DataLoader, torch.utils.data.Dataset],
    num_samples: int = 6,
    device: Optional[Union[str, torch.device]] = None,
    ignore_index: int = 255,
    colormap: Optional[np.ndarray] = None,
    denorm: bool = True,
    save_dir: Optional[str] = None,
    exp_name: str = "colab_viz"
) -> List[str]:
    """
    Esegue forward con PIDNet e salva triplet (Input, GT, Pred) in:
      PIDNet/output/<exp_name>/preds_<timestamp>/
    Ritorna i path dei file salvati.
    """
    if save_dir is None:
        out_root = _default_output_root()
        save_dir = _ensure_dir(os.path.join(out_root, exp_name, f"preds_{_timestamp()}"))
    else:
        save_dir = _ensure_dir(save_dir)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    def _iter_stream(d):
        # Dataset (no batch) -> emula batch=1
        if hasattr(d, "__len__") and hasattr(d, "__getitem__") and not hasattr(d, "batch_size"):
            for i in range(len(d)):
                yield (d[i],)
        else:
            # DataLoader
            yield from d

    saved_paths = []
    shown = 0
    for batch in _iter_stream(data):
        # normalizza in (images, masks, names)
        if len(batch) == 1:
            item = _smart_unpack_item(batch[0])
            images_t, masks_t, names = item[0], item[1], item[2]
            images_t = images_t.unsqueeze(0)
            masks_t  = masks_t.unsqueeze(0) if hasattr(masks_t, "unsqueeze") else torch.as_tensor(masks_t)[None, ...]
            names = [names] if names is not None else None
        else:
            images_t, masks_t, names = _smart_unpack_batch(batch)

        images = images_t.to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        preds = outputs.argmax(dim=1).detach().cpu().numpy()     # (B,H,W)

        imgs_np = images.detach().cpu().numpy()                   # (B,3,H,W)
        gts_np  = masks_t.detach().cpu().numpy() if hasattr(masks_t, "detach") else np.array(masks_t)

        B = preds.shape[0]
        for b in range(B):
            if shown >= num_samples:
                print(f"[save_predictions_pidnet] Saved {shown} triplets to: {save_dir}")
                return saved_paths

            img_vis = denormalize_img(imgs_np[b]) if denorm else np.transpose(imgs_np[b], (1,2,0))
            gt_vis  = decode_segmap(gts_np[b],  colormap=colormap, ignore_index=ignore_index)
            pr_vis  = decode_segmap(preds[b],   colormap=colormap, ignore_index=ignore_index)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img_vis); axs[0].set_title('Input');         axs[0].axis('off')
            axs[1].imshow(gt_vis);  axs[1].set_title('Ground Truth');  axs[1].axis('off')
            axs[2].imshow(pr_vis);  axs[2].set_title('Prediction');    axs[2].axis('off')
            plt.tight_layout()

            name_b = (names[b] if names is not None else f"sample_{shown}")
            safe_name = str(name_b).replace("/", "_")
            fpath = os.path.join(save_dir, f"{shown:03d}_{safe_name}.png")
            fig.savefig(fpath, dpi=120)
            plt.close(fig)

            saved_paths.append(fpath)
            shown += 1

    print(f"[save_predictions_pidnet] Saved {shown} triplets to: {save_dir}")
    return saved_paths


# -----------------------------------------------------------------------------
# (Opzionale) Loader robusto da checkpoint
# -----------------------------------------------------------------------------
def load_pidnet_from_checkpoint(
    build_model_fn,
    cfg,
    ckpt_path: str,
    device: Optional[Union[str, torch.device]] = None
) -> torch.nn.Module:
    """Carica PIDNet da checkpoint gestendo prefissi 'module.' e 'model.'."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model_fn(cfg).to(device)
    raw = torch.load(ckpt_path, map_location=device)

    def _try_load(sd):
        new_sd = {}
        for k, v in sd.items():
            k2 = k
            if k2.startswith('module.'):
                k2 = k2[len('module.'):]
            if k2.startswith('model.'):
                k2 = k2[len('model.'):]
            new_sd[k2] = v
        model.load_state_dict(new_sd, strict=False)

    if isinstance(raw, dict) and 'state_dict' in raw:
        _try_load(raw['state_dict'])
    elif isinstance(raw, dict):
        _try_load(raw)
    else:
        raise ValueError("Checkpoint non riconosciuto: atteso dict con 'state_dict' o state_dict flat.")
    model.eval()
    return model
