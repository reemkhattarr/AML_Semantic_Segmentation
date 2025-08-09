import os

def make_lst(split, out_path, relative_prefix, img_dir, mask_dir):
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for file in files:
            img_rel = os.path.join(relative_prefix, "images_png", file)
            mask_rel = os.path.join(relative_prefix, "masks_png", file)
            f.write(f"{img_rel} {mask_rel}\n")
    print(f"✅ Creato {out_path} con {len(files)} elementi.")

# Base path corretti
dataset_base = "/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA"
list_base = "/content/drive/MyDrive/AML_Semantic_Segmentation/PIDNet/data/list/loveda"

# TRAIN - Rural
make_lst(
    split="Train",
    out_path=os.path.join(list_base, "Train/Rural/train.lst"),
    relative_prefix="Train/Rural",  # <-- CORRETTO QUI
    img_dir=os.path.join(dataset_base, "Train/Rural/images_png"),
    mask_dir=os.path.join(dataset_base, "Train/Rural/masks_png")
)

# VAL - Rural
make_lst(
    split="Val",
    out_path=os.path.join(list_base, "Val/Rural/val.lst"),
    relative_prefix="Val/Rural",  # <-- CORRETTO QUI
    img_dir=os.path.join(dataset_base, "Val/Rural/images_png"),
    mask_dir=os.path.join(dataset_base, "Val/Rural/masks_png")
)
