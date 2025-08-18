import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import yaml
import _init_paths


# LoveDA 7-class color palette
LOVEDA_COLORMAP = np.array([
    [255, 255, 255],  # 0: Background
    [255, 0, 0],      # 1: Building
    [0, 255, 0],      # 2: Road
    [0, 0, 255],      # 3: Water
    [255, 255, 0],    # 4: Barren
    [0, 255, 255],    # 5: Forest
    [255, 0, 255],    # 6: Agriculture
], dtype=np.uint8)

def decode_segmap(mask, colormap=LOVEDA_COLORMAP, ignore_index=None):
    if mask.ndim == 2:
        mask = mask[None, ...]
    N, H, W = mask.shape
    color_masks = np.zeros((N, H, W, 3), dtype=np.uint8)
    for i in range(N):
        for cls_idx, color in enumerate(colormap):
            color_masks[i][mask[i] == cls_idx] = color
        if ignore_index is not None:
            color_masks[i][mask[i] == ignore_index] = [0, 0, 0]
    if color_masks.shape[0] == 1:
        return color_masks[0]
    return color_masks

def visualize(config_path, checkpoint_path, split='val', output_dir='visualizations', num_samples=10):
    # Load config
    from configs import config as cfg
    from configs import update_config
    
    update_config(cfg, argparse.Namespace(cfg=config_path, opts=[]))
    config = cfg  


    import models
    import datasets

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset instantiation (as in your train/eval scripts)
    list_path = config['DATASET']['TEST_SET'] if split == 'val' else config['DATASET']['TRAIN_SET']
    dataset = getattr(datasets, config['DATASET']['DATASET'])(
        root=config['DATASET']['ROOT'],
        list_path=list_path,
        num_classes=config['DATASET']['NUM_CLASSES'],
        multi_scale=False,
        flip=False,
        ignore_label=config['TRAIN']['IGNORE_LABEL'],
        base_size=config['TEST']['BASE_SIZE'],
        crop_size=tuple(config['TEST']['IMAGE_SIZE'])
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model instantiation
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict({k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items() if (k[6:] if k.startswith('model.') else k) in model.state_dict()})
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for batch in loader:
            image, label, edge, size, name = batch
            image = image.to(device)
            # PIDNet may return a list (if augment=True) or tensor (if augment=False)
            output = model(image)
            if isinstance(output, list):
                output = output[1]  # main output
            pred = output.argmax(dim=1).cpu().numpy()[0]
            img = image.cpu().numpy()[0].transpose(1,2,0)
            # Unnormalize for visualization
            mean = np.array(config['TRAIN'].get('MEAN', [0.485, 0.456, 0.406]))
            std = np.array(config['TRAIN'].get('STD', [0.229, 0.224, 0.225]))
            img = (img * std + mean)
            img = np.clip(img, 0, 1)
            gt = label.cpu().numpy()[0]
            pred_vis = decode_segmap(pred)
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
            plt.savefig(os.path.join(output_dir, f'{name[0]}.png'))
            plt.close(fig)
            count += 1
            if count >= num_samples:
                break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--output_dir', type=str, default='visualizations')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    visualize(args.config, args.checkpoint, args.split, args.output_dir, args.num_samples)



# run after training
# python visualization.py --config pidnet_loveda_urban.yaml --checkpoint output/best.pt --split val --output_dir visualizations --num_samples 10
