import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LoveDA(Dataset):
    def __init__(self, root, list_path, num_classes=7, multi_scale=False,
                 flip=False, ignore_label=255, base_size=512, crop_size=512, scale_factor=16):
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.multi_scale = multi_scale
        self.flip = flip
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_factor = scale_factor

        # read image and mask paths
        with open(list_path, 'r') as f:
            self.img_ids = [line.strip() for line in f.readlines()]

        self.files = []
        for name in self.img_ids:
            image_file = os.path.join(self.root, 'images', name + '.png')
            label_file = os.path.join(self.root, 'masks', name + '.png')
            self.files.append({
                "img": image_file,
                "label": label_file,
                "name": name
            })

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        image = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        label = label.resize((self.crop_size, self.crop_size), Image.NEAREST)

        image = self.to_tensor(image)
        image = self.normalize(image)
        label = np.array(label).astype('int32')

        return image, label, datafiles["name"]
