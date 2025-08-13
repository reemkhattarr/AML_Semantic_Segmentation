import os
import random
import yaml
from DeepLabV2.src.utils import logger
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
        split=config['data']['train_split'],
        image_dir=config['data']['image_dir'],
        mask_dir=config['data']['mask_dir'],
        input_size=tuple(config['data']['input_size']),
        transforms=None  # No augmentations for step 2a
    )
    val_dataset = LoveDADataset(
        root=config['data']['root'],
        split=config['data']['val_split'],
        image_dir=config['data']['image_dir'],
        mask_dir=config['data']['mask_dir'],
        input_size=tuple(config['data']['input_size']),
        transforms=None
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
    
    '''
    # matches deeplabv2's original training strategy
    if hasattr(model, 'optim_parameters'):
        param_groups = model.optim_parameters(
            base_lr=config['train']['lr'],
            head_lr_multiplier=10.0
        )
        optimizer = SGD(param_groups,
                    momentum=config['train']['momentum'],
                    weight_decay=config['train']['weight_decay'])
    else:
        optimizer = SGD(model.parameters(),
                    lr=config['train']['lr'],
                    momentum=config['train']['momentum'],
                    weight_decay=config['train']['weight_decay'])
    '''
    

    # weights calculated on the rural training set
    weights = [0.19429056, 1.8521928, 2.52567139, 0.57098341, 1.50308052, 0.21506205, 0.13871927]
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    # Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                    ignore_index=config['data']['ignore_index'])

    # Experiment tracking
    if WANDB_AVAILABLE and config.get('wandb', {}).get('enable', False):
        wandb.init(project=config['wandb']['project'], config=config)

    best_miou = 0.0
    for epoch in range(config['train']['epochs']):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}"):
            print("Batch images shape:", images.shape)
            print("Batch masks unique:", torch.unique(masks))
            print("Batch masks min/max:", masks.min().item(), masks.max().item())

            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Loss: {avg_loss:.4f}")
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        miou_metric = MeanIoU(num_classes=config['model']['num_classes'], ignore_index=config['data']['ignore_index'])
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                print("Output logits shape:", outputs.shape)   # Should be (N, num_classes, H, W)
                print("Pred argmax unique classes:", torch.unique(outputs.argmax(dim=1)))
                preds = outputs.argmax(dim=1)
                miou_metric.update(preds.cpu(), masks.cpu())
        
        '''
        miou = miou_metric.compute()
        logger.info(f"Epoch {epoch+1}: Val mIoU: {miou:.4f}")
        '''
        
        
        miou, per_class_iou = miou_metric.compute(return_per_class=True)
        logger.info(f"Epoch {epoch+1}: Val mIoU: {miou:.4f}")

        for idx, class_iou in enumerate(per_class_iou):
            logger.info(f"    Class {idx} IoU: {class_iou:.4f}")

        print(f"Val mIoU: {miou:.4f}")
        for idx, class_iou in enumerate(per_class_iou):
            print(f"    Class {idx} IoU: {class_iou:.4f}")


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
