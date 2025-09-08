# pem/data/register_loveda_areas.py
from detectron2.data import DatasetCatalog, MetadataCatalog
import os, glob, cv2, numpy as np

LOVEDA_CLASSES = ["background","building","road","water","barren","forest","agriculture"]

def remap_loveda_mask(mask: np.ndarray) -> np.ndarray:
    # LoveDA: 0 = no-data/ignore, 1..7 = classi  →  rimappa a: 255 = ignore, 0..6 = classi
    out = np.full(mask.shape, 255, np.uint8)
    m = mask > 0
    out[m] = (mask[m].astype(np.uint8) - 1)
    return out

def build_and_cache_remapped_mask(src_path: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if not os.path.exists(dst_path):
        m = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {src_path}")
        if m.ndim == 3:   # safety per PNG palettizzati/colori
            m = m[:, :, 0]
        m = remap_loveda_mask(m.astype("uint8"))
        cv2.imwrite(dst_path, m)
    return dst_path

def _scan(img_root: str, src_mask_root: str, split: str, area: str | None, remap_cache_root: str | None):
    recs = []
    areas = [area] if area is not None else ["Urban", "Rural"]
    for a in areas:
        img_dir  = os.path.join(img_root,  split, a, "images_png")
        src_mdir = os.path.join(src_mask_root, split, a, "masks_png")
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        for img_path in img_paths:
            fname = os.path.basename(img_path)
            src_mask = os.path.join(src_mdir, fname)
            if not os.path.exists(src_mask):
                raise FileNotFoundError(f"Mask non trovata: {src_mask}")
            if remap_cache_root is not None:
                rel = os.path.join(split, a, "masks_png", fname)
                dst_mask = os.path.join(remap_cache_root, rel)
                sem_path = build_and_cache_remapped_mask(src_mask, dst_mask)
            else:
                sem_path = src_mask  # usa maschere “as-is”
            h, w = cv2.imread(img_path).shape[:2]
            recs.append({
                "file_name": img_path,
                "sem_seg_file_name": sem_path,
                "height": h,
                "width":  w,
            })
    return recs

def register_loveda_areas(
    img_root: str,
    mask_root: str | None = None,          # se None, usa img_root
    remap_cache_root: str | None = None,   # se settato ⇒ rimappa+cache
    ignore: int | None = None,             # default: 255 se remap, altrimenti 0
    make_combined: bool = True,
    warm_cache: bool = False               # se True, costruisce subito tutte le cache
):
    src_mask_root = mask_root if mask_root is not None else img_root
    if ignore is None:
        ignore = 255 if remap_cache_root is not None else 0

    def _reg(name, split, area):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        DatasetCatalog.register(
            name,
            # cattura gli argomenti nei default della lambda (no late-binding)
            lambda s=split, a=area, ir=img_root, mr=src_mask_root, rc=remap_cache_root:
                _scan(ir, mr, s, a, rc)
        )
        MetadataCatalog.get(name).set(
            stuff_classes=LOVEDA_CLASSES,
            ignore_label=ignore,
            evaluator_type="sem_seg",
        )

    # # combinati
    # if make_combined:
    #     for split in ["Train", "Val"]:
    #         _reg(f"loveda_{split.lower()}", split, area=None)

    # per-area
    for split in ["Train", "Val"]:
        for area in ["Urban", "Rural"]:
            _reg(f"loveda_{split.lower()}_{area.lower()}", split, area)

    # opzionale: warm-cache immediato (genera Train/Val/Urban/Rural)
    if warm_cache and remap_cache_root is not None:
        _ = _scan(img_root, src_mask_root, "Train", None, remap_cache_root)
        _ = _scan(img_root, src_mask_root, "Val",   None, remap_cache_root)
