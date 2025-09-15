# train_fastai_timm.py (CPU-optimized with AUROC/AUPRC at test stage only)

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import timm
from fastai.vision.all import *
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision

# ---------------- CONFIG ----------------
DATA_ROOT = Path("../COVID-19_Radiography_Dataset")
MODEL_NAME = "mobilenetv3_small_100"   # ✅ light CPU model
BATCH_SIZE = 16
IMAGE_SIZE = 160
EPOCHS = 5
SEED = 42
NUM_WORKERS = 0                       # ✅ safer on Windows
CHECKPOINT_DIR = Path("../models")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

set_seed(SEED, reproducible=True)

# ---------------- Dataset ----------------
def collect_dataset(root: Path, classes=None):
    samples = []
    if classes is None:
        classes = [p.name for p in root.iterdir() if p.is_dir()]
    for c in classes:
        class_folder = root / c / "images"
        if not class_folder.is_dir():
            class_folder = root / c
        if not class_folder.is_dir():
            continue
        for img in class_folder.iterdir():
            if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                samples.append({"path": str(img.resolve()), "label": c})
    return pd.DataFrame(samples)

print("Collecting dataset from:", DATA_ROOT)
df = collect_dataset(DATA_ROOT)
print("Total images found:", len(df))
print(df['label'].value_counts())

# ---------------- Splits ----------------
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
train_df, val_df  = train_test_split(train_df, test_size=0.15, stratify=train_df["label"], random_state=SEED)
print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

combined = pd.concat([train_df.reset_index(drop=True), val_df.reset_index(drop=True)], ignore_index=True)
val_start = len(train_df)
splitter = IndexSplitter(list(range(val_start, val_start + len(val_df))))

# ---------------- DataBlock ----------------
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader("path"),
    get_y=ColReader("label"),
    splitter=splitter,
    item_tfms=Resize(224),
    batch_tfms=[
        *aug_transforms(size=IMAGE_SIZE, flip_vert=False, max_rotate=5, max_zoom=1.05, max_lighting=0.1),
        Normalize.from_stats(*imagenet_stats)
    ],
)

dls = dblock.dataloaders(combined, bs=BATCH_SIZE, num_workers=NUM_WORKERS)

print("Classes:", dls.vocab)
n_classes = len(dls.vocab)

# ---------------- Model ----------------
def create_timm_model(arch_name, n_out, pretrained=True):
    return timm.create_model(arch_name, pretrained=pretrained, num_classes=n_out)

model = create_timm_model(MODEL_NAME, n_classes, pretrained=True)

# ---------------- Learner ----------------
learn = Learner(
    dls, model,
    loss_func=CrossEntropyLossFlat(),
    metrics=[accuracy],
    cbs=[
        SaveModelCallback(monitor='accuracy', fname=f'{MODEL_NAME}_best'),
        EarlyStoppingCallback(monitor='accuracy', patience=3),
    ]
)

print("🚀 Starting training on CPU...")
learn.fine_tune(EPOCHS, base_lr=1e-3)

# ---------------- Save ----------------
export_path = CHECKPOINT_DIR / f"{MODEL_NAME}_cpu_fastai.pkl"
learn.export(export_path)
print("✅ Model exported to:", export_path)

# # ---------------- Evaluate ----------------
# print("\n📊 Evaluation on test set...")

# # build test dataloader
# # build test dataloader WITH labels
# test_items = [(row["path"], row["label"]) for _, row in test_df.iterrows()]
# test_dl = learn.dls.test_dl(test_items, bs=BATCH_SIZE, num_workers=NUM_WORKERS)

# preds, targs = learn.get_preds(dl=test_dl)

# # predictions
# preds, targs = learn.get_preds(dl=test_dl)
# pred_labels = preds.argmax(dim=1)
# true_labels = targs

# # sklearn classification report
# print(classification_report(true_labels, pred_labels, target_names=dls.vocab))

# # AUROC + AUPRC on test set
# roc_auc = MulticlassAUROC(num_classes=n_classes, average="macro")(preds, true_labels)
# auprc   = MulticlassAveragePrecision(num_classes=n_classes, average="macro")(preds, true_labels)

# print(f"Test AUROC: {roc_auc:.4f}")
# print(f"Test AUPRC: {auprc:.4f}")

# # confusion matrix
# cm = confusion_matrix(true_labels, pred_labels)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d',
#             xticklabels=dls.vocab, yticklabels=dls.vocab, cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix (Test Set)")
# plt.show()