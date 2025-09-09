# train_fastai_timm.py (CPU-optimized)

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import timm
from fastai.vision.all import *

# ---------------- CONFIG ----------------
DATA_ROOT = Path("../COVID-19_Radiography_Dataset")
MODEL_NAME = "mobilenetv3_small_100"   # ✅ light model for CPU
BATCH_SIZE = 16                        # ✅ smaller batch size for CPU
IMAGE_SIZE = 160                       # ✅ smaller image resolution
EPOCHS = 5
SEED = 42
NUM_WORKERS = 0                        # ✅ safer for Windows CPU
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
    item_tfms=Resize(224),  # fast initial resize
    batch_tfms=[
        *aug_transforms(size=IMAGE_SIZE, flip_vert=False, max_rotate=5, max_zoom=1.05, max_lighting=0.1),
        Normalize.from_stats(*imagenet_stats)
    ],
)

dls = dblock.dataloaders(combined, bs=BATCH_SIZE, num_workers=NUM_WORKERS)

print("Classes:", dls.vocab)
n_classes = len(dls.vocab)

roc_auc = MultilabelAUROC(num_labels=n_classes, average="macro")
# AUPRC (macro, average precision)
auprc = MultilabelAveragePrecision(num_labels=n_classes, average="macro")
# ---------------- Model ----------------
def create_timm_model(arch_name, n_out, pretrained=True):
    return timm.create_model(arch_name, pretrained=pretrained, num_classes=n_out)

model = create_timm_model(MODEL_NAME, n_classes, pretrained=True)
class TorchMetricWrapper(Callback):
    def __init__(self, metric, name):
        self.metric = metric
        self.name = name

    def before_fit(self): self.metric.reset()
    def before_epoch(self): self.metric.reset()
    def after_batch(self):
        if not self.training:
            preds = self.learn.pred
            targs = self.learn.y
            self.metric.update(preds.cpu(), targs.cpu())
    def after_epoch(self):
        val = self.metric.compute().item()
        self.learn.recorder.log.append(val)
        print(f"{self.name}: {val:.4f}")
        self.metric.reset()

# ---------------- Learner ----------------
learn = Learner(
    dls, model,
    loss_func=CrossEntropyLossFlat(),
    metrics=[accuracy],
    cbs=[SaveModelCallback(monitor='accuracy', fname=f'{MODEL_NAME}_best'),
        EarlyStoppingCallback(monitor='accuracy', patience=3),
        TorchMetricWrapper(roc_auc, "val_auroc"),
        TorchMetricWrapper(auprc, "val_auprc"),
         ]
)

print("Starting training on CPU...")
learn.fine_tune(EPOCHS, base_lr=1e-3)

# ---------------- Save ----------------
export_path = CHECKPOINT_DIR / f"{MODEL_NAME}_cpu_fastai.pkl"
learn.export(export_path)
print("✅ Model exported to:", export_path)

# ---------------- Evaluate ----------------
test_files = list(test_df['path'].values)
test_dl = learn.dls.test_dl(test_files, bs=BATCH_SIZE, num_workers=NUM_WORKERS)

preds, targs = learn.get_preds(dl=test_dl)
pred_labels = preds.argmax(dim=1).numpy()
true_labels = targs.numpy()

print("\n📊 Classification report (test set):")
print(classification_report(true_labels, pred_labels, target_names=dls.vocab))

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=dls.vocab, yticklabels=dls.vocab, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()
