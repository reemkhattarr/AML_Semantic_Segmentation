# pem/data/register_loveda_areas_raw.py
from detectron2.data import DatasetCatalog, MetadataCatalog
import os, glob, cv2

LOVEDA_CLASSES = ["background","building","road","water","barren","forest","agriculture"]

def _scan_raw(root: str, split: str, area: str | None):
    recs = []
    areas = [area] if area is not None else ["Urban", "Rural"]
    for a in areas:
        img_dir  = os.path.join(root, split, a, "images_png")
        mask_dir = os.path.join(root, split, a, "masks_png")
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        for img_path in img_paths:
            fname    = os.path.basename(img_path)
            mask_png = os.path.join(mask_dir, fname)  # MASCHERA ORIGINALE (0..7, con 0=ignore)
            if not os.path.exists(mask_png):
                raise FileNotFoundError(f"Mask non trovata: {mask_png}")
            h, w = cv2.imread(img_path).shape[:2]
            recs.append({
                "file_name": img_path,
                "sem_seg_file_name": mask_png,
                "height": h,
                "width":  w,
            })
    return recs

def register_loveda_areas_raw(root: str, make_combined: bool = False):
    """
    Registra:
      - loveda_train_urban_raw, loveda_train_rural_raw
      - loveda_val_urban_raw,   loveda_val_rural_raw
    Se make_combined=True aggiunge:
      - loveda_train_raw, loveda_val_raw
    """
    def _reg(name, split, area):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        DatasetCatalog.register(name, lambda s=split, a=area, r=root: _scan_raw(r, s, a))
        # IMPORTANTE: per i PNG “raw” di LoveDA l'ignore è 0
        MetadataCatalog.get(name).set(
            stuff_classes=LOVEDA_CLASSES,
            ignore_label=0,            # 0 = ignore nei file originali LoveDA
            evaluator_type="sem_seg",
        )

    # per-area
    for split in ["Train", "Val"]:
        for area in ["Urban", "Rural"]:
            _reg(f"loveda_{split.lower()}_{area.lower()}_raw", split, area)

    # combinati (opzionali)
    if make_combined:
        for split in ["Train", "Val"]:
            _reg(f"loveda_{split.lower()}_raw", split, area=None)
