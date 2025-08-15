# ------------------------------------------------------------------------------
# LoveDA Dataset for PIDNet (compatible with your repo)
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
from PIL import Image
import torch
from .base_dataset import BaseDataset

try:
    import albumentations as A
except ImportError:
    A = None

# You can adjust these based on your needs
LOVEDA_CLASS_WEIGHTS = [0.000000, 0.116411, 0.266041, 0.607794, 1.511413, 0.745507, 0.712438, 3.040396]

class LoveDA(BaseDataset):
    def __init__(self,
                root,
                list_path,
                num_classes=7,
                multi_scale=False,
                flip=False,
                ignore_label=-1,
                base_size=720,
                crop_size=(720, 720),
                scale_factor=16,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                bd_dilate_size=4,
                use_augmentation=False,
                aug_prob=0.5,
                transform=None):

        super(LoveDA, self).__init__(
            ignore_label, base_size, crop_size, scale_factor, mean, std
        )

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.ignore_label = ignore_label
        self.bd_dilate_size = bd_dilate_size

        self.color_list = [[0, 0, 0],         # 0: Ignore
                            [255, 255, 255],  # 1: Background
                            [255, 0, 0],      # 2: Building
                            [0, 255, 0],      # 3: Road
                            [0, 0, 255],      # 4: Water
                            [255, 255, 0],    # 5: Barren
                            [0, 255, 255],    # 6: Forest
                            [255, 0, 255]]    # 7: Agriculture

        self.img_list = [line.strip().split() for line in open(os.path.join(root, list_path))]
        self.files = self.read_files()

        # Class weights for loss
        self.class_weights = torch.tensor(LOVEDA_CLASS_WEIGHTS, dtype=torch.float32)
        


    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
        return files
    
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2]) * self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2) == 3] = i

        return label.astype(np.uint8)
    

    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,))
        for i, v in enumerate(self.color_list):
            color_map[label == i] = self.color_list[i]

        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, 'LoveDA', item["img"]), cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.root, 'LoveDA', item["label"]), cv2.IMREAD_GRAYSCALE)
        size = image.shape

        # Convert to model's ignore label
        label = np.where(label == 255, self.ignore_label, label)

        # Generate edge map, and apply further transforms as required by BaseDataset
        image, label, edge = self.gen_sample(
            image, label, self.multi_scale, self.flip, edge_size=self.bd_dilate_size
        )

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))


def generate_lst(path_images, path_labels, output_path):
    '''
    Generate a list of image and label paths.
    '''
    images = sorted(os.listdir(path_images))
    labels = sorted(os.listdir(path_labels))
    
    with open(output_path, 'w') as f:
        for image, label in zip(images, labels):
            path_image = os.path.join(path_images, image).replace("\\", "/")
            path_label = os.path.join(path_labels, label).replace("\\", "/")
            f.write(f"{path_image}\t{path_label}\n")