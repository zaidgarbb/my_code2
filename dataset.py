
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LabeledDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

        if augment:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(p=0.3),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 0).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"]

class UnlabeledDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        transformed = self.transform(image=image)
        return transformed["image"]
