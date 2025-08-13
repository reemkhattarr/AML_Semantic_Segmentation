import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class LoveDADataset(Dataset):
    def __init__(self, root, split, image_dir, mask_dir, input_size=(720, 720), transforms=None):
        self.root = root
        self.split = split  # e.g., train/rural or val/rural
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.transforms = transforms

        image_root = os.path.join(root, split, image_dir)
        mask_root = os.path.join(root, split, mask_dir)

        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.png')])
        self.masks = sorted([os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.endswith('.png')])
        assert len(self.images) == len(self.masks), "Image/mask count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load and preprocess the image
        image = Image.open(self.images[idx]).convert('RGB').resize(self.input_size)
        mask = Image.open(self.masks[idx]).resize(self.input_size, resample=Image.NEAREST)
        print("Raw mask unique:", np.unique(mask))
        
        # Convert to numpy arrays
        image = np.asarray(image).astype(np.float32) / 255.0
        mask = np.asarray(mask).astype(np.int64)
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Subtract 1 from the mask to make it 0-indexed
        mask = mask - 1
        print("Shifted mask unique:", np.unique(mask))  # Should be [0, ..., 6]
        print("Mask min/max shape:", mask.min(), mask.max(), mask.shape)

        # Convert to torch.Tensor and permute to (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        mask = torch.from_numpy(mask).long()  # (H, W)
        
        # Apply any additional transformations
        if self.transforms:
            image, mask = self.transforms(image, mask)
        
        return image, mask
