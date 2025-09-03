import os
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

COVID_CLASSES=["Normal","Lung_Opacity", "Viral Pneumonia", "COVID"]
class CovidXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_options=False):
        self.samples = []
        self.transform = transform
        classes = COVID_CLASSES.copy()
        if include_options:
            classes.append("Lung_Opacity")
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            paths = glob(os.path.join(root_dir, c, "*.png"))
            for p in paths:
                self.samples.append((p, self.class_to_idx[c]))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img=Image.open(path).convert("RGB")
        img=np.array(img)
        if self.transform:
            img=self.transform(image=img)["image"]
        return img, label
