import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from fastai.vision.all import load_learner
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision

# ---------------- Logging ----------------
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "evaluate.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------- Config ----------------
DATA_ROOT = Path("../COVID-19_Radiography_Dataset")
MODEL_PATH = Path("../models/mobilenetv3_small_100_cpu_fastai.pkl")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0

# ---------------- Dataset ----------------
logger.info(f"Collecting dataset from {DATA_ROOT}")
samples = []
classes = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
for c in classes:
    class_folder = DATA_ROOT / c / "images"
    if not class_folder.is_dir():
        class_folder = DATA_ROOT / c
    for img in class_folder.iterdir():
        if img.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            samples.append({"path": str(img.resolve()), "label": c})
df = pd.DataFrame(samples)

_, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
logger.info(f"Test size: {len(test_df)}")

# ---------------- Load Model ----------------
learn = load_learner(MODEL_PATH)
logger.info(f"Model loaded successfully from {MODEL_PATH}")

# ---------------- Evaluation ----------------
test_files = list(test_df["path"].values)
test_dl = learn.dls.test_dl(test_files, bs=BATCH_SIZE, num_workers=NUM_WORKERS)

preds, _ = learn.get_preds(dl=test_dl)

true_labels = np.array([learn.dls.vocab.o2i[label] for label in test_df["label"].values])
pred_labels = preds.argmax(dim=1).cpu().numpy()

# Classification report
logger.info("Classification Report (Test Set):")
logger.info("\n" + classification_report(true_labels, pred_labels, target_names=learn.dls.vocab))

# AUROC + AUPRC
roc_auc = MulticlassAUROC(num_classes=len(learn.dls.vocab), average="macro")(preds, torch.tensor(true_labels)).item()
auprc = MulticlassAveragePrecision(num_classes=len(learn.dls.vocab), average="macro")(preds, torch.tensor(true_labels)).item()

logger.info(f"Test AUROC: {roc_auc:.4f}")
logger.info(f"Test AUPRC: {auprc:.4f}")

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=learn.dls.vocab, yticklabels=learn.dls.vocab, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path)
logger.info(f"Confusion matrix saved to {cm_path}")

# ---------------- Grad-CAM (next milestone) ----------------
logger.info("Grad-CAM integration will be added next.")
