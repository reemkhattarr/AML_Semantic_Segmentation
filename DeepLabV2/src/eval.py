import torch
import time
from torch.utils.data import DataLoader
from deeplabv2 import get_deeplabv2_model
from src.dataset import LoveDADataset
from src.utils.metrics import MeanIoU, compute_flops_and_params 

def evaluate(config, checkpoint_path, split='val'):
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

    miou_metric = MeanIoU(num_classes=config['model']['num_classes'], ignore_index=config['data']['ignore_index'])

    # Latency, FLOPs, Params
    images, _ = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        start = time.time()
        _ = model(images)
        latency = time.time() - start
    flops, params = compute_flops_and_params(model, images.shape)

    # mIoU
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            miou_metric.update(preds.cpu(), masks.cpu())
    miou, per_class_iou = miou_metric.compute(return_per_class=True)

    print(f"Latency (1 image): {latency*1000:.2f} ms")
    print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
    print(f"Params: {params/1e6:.2f} M")
    print(f"mIoU: {miou:.4f}")
    print("Per-class IoU:", per_class_iou)

if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    evaluate(config, args.checkpoint, args.split)
