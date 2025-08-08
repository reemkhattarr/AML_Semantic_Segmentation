import os
import random
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import SGD
from tqdm import tqdm

from deeplabv2 import get_deeplabv2_model
from src.dataset import LoveDADataset
from src.utils.metrics import MeanIoU  # You should implement this
from src.utils.logger import setup_logger  # Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path: str = "configs/train_deeplabv2_loveda.yaml"):
    config = load_config(config_path)
    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger()

    # Dataset
    train_dataset = LoveDADataset(
        root=config['data']['root'],
        split='train',
        transform=config['data']['train_transform']
    )
    val_dataset = LoveDADataset(
        root=config['data']['root'],
        split='val',
        transform=config['data']['val_transform']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model
    model = get_deeplabv2_model(
        num_classes=config['model']['num_classes'],
        pretrained_backbone=True,
        freeze_bn=config['model']['freeze_bn']
    ).to(device)

    # Optimizer (1x/10x LR per DeepLabV2 paper [[17]])
    optimizer = SGD(
        model.optim_parameters(config['train']['lr']),
        momentum=config['train']['momentum'],
        weight_decay=config['train']['weight_decay']
    )

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=config['data']['ignore_index'])

    # Experiment tracking
    if WANDB_AVAILABLE and config.get('wandb', {}).get('enable', False):
        wandb.init(project=config['wandb']['project'], config=config)

    best_miou = 0.0
    for epoch in range(config['train']['epochs']):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        miou_metric = MeanIoU(num_classes=config['model']['num_classes'], ignore_index=config['data']['ignore_index'])
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                miou_metric.update(preds.cpu(), masks.cpu())
        miou = miou_metric.compute()
        logger.info(f"Epoch {epoch+1}: Val mIoU: {miou:.4f}")

        # Save best model
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), os.path.join(config['train']['output_dir'], 'best_model.pth'))

        # Log to wandb
        if WANDB_AVAILABLE and config.get('wandb', {}).get('enable', False):
            wandb.log({'train_loss': avg_loss, 'val_miou': miou, 'epoch': epoch+1})

    logger.info(f"Training complete. Best Val mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/train_deeplabv2_loveda.yaml")
    args = parser.parse_args()
    main(args.config)
