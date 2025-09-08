import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

# Absolute import of LoveDA dataset class
from datasets.loveda import LoveDA

# Color map from LoveDA (including ignore)
LOVEDA_COLORMAP = np.array([
    [0, 0, 0],         # 0: Ignore
    [255, 255, 255],   # 1: Background
    [255, 0, 0],       # 2: Building
    [0, 255, 0],       # 3: Road
    [0, 0, 255],       # 4: Water
    [255, 255, 0],     # 5: Barren
    [0, 255, 255],     # 6: Forest
    [255, 0, 255],     # 7: Agriculture
], dtype=np.uint8)

# Softer, more natural color map for LoveDA classes (including ignore)
LOVEDA_COLORMAP = np.array([
    [0, 0, 0],           # 0: Ignore (black)
    [220, 220, 220],     # 1: Background (light gray)
    [178, 34, 34],       # 2: Building (firebrick red)
    [128, 128, 128],     # 3: Road (medium gray)
    [70, 130, 180],      # 4: Water (steel blue)
    [210, 180, 140],     # 5: Barren (tan/sandy)
    [34, 139, 34],       # 6: Forest (forest green)
    [154, 205, 50],      # 7: Agriculture (yellow green)
], dtype=np.uint8)

def remap_mask_for_visualization(mask):
    # inverse mapping: 0->1, 1->2, ..., 6->7, 255->0 (ignore)
    inv_map = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 255:0}
    new_mask = np.zeros_like(mask)
    for k, v in inv_map.items():
        new_mask[mask == k] = v
    return new_mask


def colorize_mask(mask):
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in enumerate(LOVEDA_COLORMAP):
        mask_rgb[mask == cls] = color
    return mask_rgb

def visualize_loveda_samples(dataset, num_samples=10):
    for i in range(num_samples):
        img, mask, edge, size, name = dataset[i]
        # img: tensor CHW, normalized
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        # Unnormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
        # mask: H x W, int
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        mask_color = colorize_mask(remap_mask_for_visualization(mask))
    
        # edge: H x W, float or int (0/1 or 0/255 or ignore_label)
        if isinstance(edge, torch.Tensor):
            edge = edge.numpy()
        # For visualization: show as white edges, black elsewhere (mask ignore_label as black)
        ignore_label = 255
        edge_vis = np.zeros_like(edge, dtype=np.uint8)
        edge_vis[edge == 1] = 255  # edge pixels as white
        edge_vis[edge == ignore_label] = 0    # ignore as black
    
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f"Image: {name}")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(mask_color)
        plt.title("Segmentation Mask")
        plt.axis("off")
        # plt.subplot(1, 3, 3)
        # plt.imshow(edge_vis, cmap='gray')
        # plt.title("Edge Map")
        # plt.axis("off")
        plt.show()



# if __name__ == "__main__":
#     # When run as a module, add project root to sys.path
#     import os
root = "/content/drive/MyDrive/AML_Semantic_Segmentation/PIDNet/data"
list_path = 'list/loveda/train_urban.lst'

list_path = 'list/loveda/train_rural.lst'

dataset_rural = LoveDA(
    root=root,
    list_path=list_path,
    num_classes=7,
    multi_scale=False,
    flip=False,  # handled by aug_type
    ignore_label=255,
    base_size=720,
    crop_size=(720, 720),
    augmentation_type=None,
    aug_prob=0.5
)

visualize_loveda_samples(dataset_rural, 20)

dataset_urban = LoveDA(
    root=root,
    list_path=list_path,
    num_classes=7,
    multi_scale=False,
    flip=False,  # handled by aug_type
    ignore_label=255,
    base_size=720,
    crop_size=(720, 720),
    augmentation_type=None,
    aug_prob=0.5
)


visualize_loveda_samples(dataset_urban, 20)



