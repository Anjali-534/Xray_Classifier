# train.py
import os
import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from sklearn.model_selection import train_test_split
import pandas as pd

from dataset_covid import CovidXrayDataset, CLASSES
from transforms import get_train_transforms, get_valid_transforms

# ---------------- CONFIG ----------------
DATA_ROOT = "../COVID-19_Radiography_Dataset"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
CHECKPOINT_DIR = "../models"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------- DATASET PREP ----------------
all_samples = []
for c in CLASSES:
    img_dir = os.path.join(DATA_ROOT, c, "images")
    if not os.path.isdir(img_dir):   # fallback if no images/ subfolder
        img_dir = os.path.join(DATA_ROOT, c)
    if not os.path.isdir(img_dir):
        continue
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    for f in files:
        all_samples.append({"path": os.path.join(img_dir, f), "label": c})

df = pd.DataFrame(all_samples)
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

train_ds = CovidXrayDataset(train_df, transform=get_train_transforms())
val_ds   = CovidXrayDataset(val_df, transform=get_valid_transforms())
test_ds  = CovidXrayDataset(test_df, transform=get_valid_transforms())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------------- MODEL ----------------
class XrayClassifier(pl.LightningModule):
    def __init__(self, num_classes=len(CLASSES), lr=LR):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.densenet121(weights="IMAGENET1K_V1")
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels)
        acc = self.acc(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels)
        acc = self.acc(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels)
        acc = self.acc(outputs, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

# ---------------- TRAINING ----------------
if __name__ == "__main__":
    model = XrayClassifier()

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",   # picks GPU if available, else CPU
        devices=1,
        precision=16,         # mixed precision (faster on GPU)
        default_root_dir=CHECKPOINT_DIR,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
