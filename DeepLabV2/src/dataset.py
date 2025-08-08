import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class LoveDADataset(Dataset):
    def __init__(self, root, split, image_dir, mask_dir, input_size=(512, 512), transforms=None):
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
        image = Image.open(self.images[idx]).convert('RGB').resize(self.input_size)
        mask = Image.open(self.masks[idx]).resize(self.input_size, resample=Image.NEAREST)
        image = np.asarray(image).astype(np.float32) / 255.0
        mask = np.asarray(mask).astype(np.int64)
        if self.transforms:
            image, mask = self.transforms(image, mask)
        return image, mask
