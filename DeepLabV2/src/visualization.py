import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from deeplabv2 import get_deeplabv2_model
from src.dataset import LoveDADataset
import numpy as np
from typing import Optional, Tuple, List, Dict

# LoveDA 7-class color palette 
LOVEDA_COLORMAP = np.array([
    [0, 0, 0],        # 0: Background
    [255, 0, 0],      # 1: Building
    [0, 255, 0],      # 2: Road
    [0, 0, 255],      # 3: Water
    [255, 255, 0],    # 4: Barren
    [0, 255, 255],    # 5: Forest
    [255, 0, 255],    # 6: Agriculture
], dtype=np.uint8)

def decode_segmap(
    mask: np.ndarray,
    colormap: Optional[np.ndarray] = None,
    ignore_index: Optional[int] = None,
    return_rgb: bool = True
) -> np.ndarray:
    """
    Converts a segmentation mask (H, W) or (N, H, W) of class indices to a color image or batch of images.
    
    Args:
        mask: np.ndarray of shape (H, W) or (N, H, W), dtype int/uint8, with class indices.
        colormap: np.ndarray of shape (num_classes, 3), dtype uint8. If None, uses LOVEDA_COLORMAP.
        ignore_index: Optional[int], if set, pixels with this value will be colored black.
        return_rgb: If True, returns RGB image(s); if False, returns BGR (for OpenCV).
    
    Returns:
        color_mask: np.ndarray of shape (H, W, 3) or (N, H, W, 3), dtype uint8.
    """
    if colormap is None:
        colormap = LOVEDA_COLORMAP
    if mask.ndim == 2:
        mask = mask[None, ...]  # Add batch dimension
    N, H, W = mask.shape
    color_masks = np.zeros((N, H, W, 3), dtype=np.uint8)
    for i in range(N):
        for cls_idx, color in enumerate(colormap):
            color_masks[i][mask[i] == cls_idx] = color
        if ignore_index is not None:
            color_masks[i][mask[i] == ignore_index] = [0, 0, 0]
    if not return_rgb:
        color_masks = color_masks[..., ::-1]  # RGB to BGR
    if color_masks.shape[0] == 1:
        return color_masks[0]
    return color_masks

# helper to get a colormap for a given dataset
def get_colormap(dataset: str) -> np.ndarray:
    """
    Returns the colormap for a given dataset.
    Extend this function as needed.
    """
    if dataset.lower() == "loveda":
        return LOVEDA_COLORMAP
    # Add more datasets here
    raise ValueError(f"Unknown dataset: {dataset}")



def visualize(config, checkpoint_path, split='val', output_dir='visualizations', num_samples=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LoveDADataset(
        root=config['data']['root'],
        split=split,
        transform=config['data']['val_transform']
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    model = get_deeplabv2_model(
        num_classes=config['model']['num_classes'],
        pretrained_backbone=False,
        freeze_bn=False
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()[0]
            img = images.cpu().numpy()[0].transpose(1,2,0)
            gt = masks.cpu().numpy()[0]
            pred_vis = decode_segmap(preds)
            gt_vis = decode_segmap(gt)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img)
            axs[0].set_title('Input')
            axs[1].imshow(gt_vis)
            axs[1].set_title('Ground Truth')
            axs[2].imshow(pred_vis)
            axs[2].set_title('Prediction')
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_{count}.png'))
            plt.close(fig)
            count += 1
            if count >= num_samples:
                break

if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--output_dir', type=str, default='visualizations')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    visualize(config, args.checkpoint, args.split, args.output_dir, args.num_samples)
