import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import cv2

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

        # Leggi i nomi dei file dalle liste .lst
        with open(list_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        self.files = []
        for line in lines:
            splits = line.split()
            if len(splits) == 2:
                img_rel_path, mask_rel_path = splits
                # Rimuove il prefisso "data/" per evitare path errati
                image_file = os.path.join(self.root, img_rel_path.replace("data/", ""))
                mask_file = os.path.join(self.root, mask_rel_path.replace("data/", ""))
                name = os.path.splitext(os.path.basename(img_rel_path))[0]
                self.files.append({
                    "img": image_file,
                    "label": mask_file,
                    "name": name
                })
            else:
                raise ValueError(f"Formato riga non valido nel .lst: {line}")



        # Trasformazioni
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # Pesi base per ogni classe (uniformi)
        self.class_weights = torch.tensor([1.0] * num_classes, dtype=torch.float32)

    def __len__(self):
        return len(self.files)

    def label_to_edge(self, label, edge_size=4):
        from scipy import ndimage
        edge = np.zeros(label.shape, dtype=np.uint8)
        for cls in np.unique(label):
            if cls in (255, -1):  # salta i pixel ignore
                continue
            mask = (label == cls).astype(np.uint8)
            if mask.max() == 0:
                continue
            dil = ndimage.grey_dilation(mask, size=(edge_size, edge_size))
            ero = ndimage.grey_erosion(mask, size=(edge_size, edge_size))
            edge |= (dil != ero)
        return edge.astype(np.uint8)


    def __getitem__(self, index):
        datafiles = self.files[index]

        # Leggi immagine e label
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        size = image.size[::-1]  # (H, W) originale

        # Resize
        if isinstance(self.crop_size, tuple):
            resize_size = self.crop_size
        else:
            resize_size = (self.crop_size, self.crop_size)

        image = image.resize(resize_size, Image.BILINEAR)
        label = label.resize(resize_size, Image.NEAREST)

        # To tensor + normalize
        image = self.to_tensor(image)
        image = self.normalize(image)

        label = np.array(label).astype('int32')

        # Rimappa l'ignore e qualsiasi valore fuori range
        label[label == 255] = self.ignore_label         # se nelle maschere l'ignore è 255
        label[(label < 0) | (label >= self.num_classes)] = self.ignore_label

        # Genera mappa bordi
        edge = self.label_to_edge(label, edge_size=4)

        # Ritorna i 5 elementi richiesti
        return image, label, edge, np.array(size), datafiles["name"]
