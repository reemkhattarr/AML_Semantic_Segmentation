# ------------------------------------------------------------------------------
# LoveDA Dataset for PIDNet (compatible with your repo)
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
from PIL import Image
import torch
from .base_dataset import BaseDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

LOVEDA_CLASS_WEIGHTS = [0.116411, 0.266041, 0.607794, 1.511413, 0.745507, 0.712438, 3.040396]

class LoveDA(BaseDataset):
    def __init__(self,
                root,
                list_path,
                num_classes=7,
                multi_scale=False,
                flip=False,
                ignore_label=255,
                base_size=720,
                crop_size=(720, 720),
                scale_factor=16,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                bd_dilate_size=4,
                augmentation_type=None,
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
        
        self.label_mapping = {0: ignore_label, 
                            1: 0, 2: 1, 
                            3: 2, 4: 3, 
                            5: 4, 6: 5, 
                            7: 6}

        # Class weights for loss
        self.class_weights = torch.tensor(LOVEDA_CLASS_WEIGHTS, dtype=torch.float32)
        
        self.augmentation = get_augmentation(augmentation_type, aug_prob)


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
    
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
    
        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root, 'LoveDA', item["img"]), cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(self.root, 'LoveDA', item["label"]), cv2.IMREAD_GRAYSCALE)
        size = image.shape

        label = self.convert_label(label)
        
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

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
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))


def get_augmentation(aug_type=None, prob=0.5):
    if aug_type is None or aug_type == 'none':
        return None
    # If it's a string, make it a list
    if isinstance(aug_type, str):
        aug_type = [aug_type]
    aug_list = []
    if 'flip' in aug_type:
        aug_list.append(A.HorizontalFlip(p=prob))
        aug_list.append(A.VerticalFlip(p=prob))
    if 'blur' in aug_type:
        aug_list.append(A.GaussianBlur(blur_limit=(3, 7), p=prob))
    if 'multiply' in aug_type:
        aug_list.append(A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=prob))
    if 'color' in aug_type:
        aug_list.append(A.ColorJitter(p=prob))
    if 'rotate' in aug_type:
        aug_list.append(A.RandomRotate90(p=prob))
    if aug_list:
        return A.Compose(aug_list)
    return None



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
            

train_urban_images_path = '/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA/Train/Urban/images_png'
train_urban_masks_path = '/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA/Train/Urban/masks_png'
train_urban_lst_path = '/content/drive/MyDrive/AML_Semantic_Segmentation/PIDNet/data/list/loveda/train_urban.lst'

val_urban_images_path = '/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA/Val/Urban/images_png'
val_urban_masks_path = '/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA/Val/Urban/masks_png'
val_urban_lst_path = '/content/drive/MyDrive/AML_Semantic_Segmentation/PIDNet/data/list/loveda/val_urban.lst'

val_rural_images_path = '/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA/Val/Rural/images_png'
val_rural_masks_path = '/content/drive/MyDrive/AML_Semantic_Segmentation/data/LoveDA/Val/Rural/masks_png'
val_rural_lst_path = '/content/drive/MyDrive/AML_Semantic_Segmentation/PIDNet/data/list/loveda/val_rural.lst'

generate_lst(train_urban_images_path, train_urban_masks_path, train_urban_lst_path)
generate_lst(val_urban_images_path, val_urban_masks_path, val_urban_lst_path)
generate_lst(val_rural_images_path, val_rural_masks_path, val_rural_lst_path)