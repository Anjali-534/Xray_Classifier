# dataset_covid.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# define class labels
CLASSES = ["Normal", "Lung_Opacity", "Viral Pneumonia", "COVID"]

class CovidXrayDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: pandas DataFrame with columns ["path", "label"]
        transform: Albumentations transform pipeline
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = np.array(img)
        label = self.class_to_idx[row["label"]]

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long)
