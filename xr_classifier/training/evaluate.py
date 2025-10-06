import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# --- New utilities ---
from calibration_utils import ModelWithTemperature, compute_ece, plot_reliability_diagram
from mc_dropout_utils import mc_dropout_predict
from gradcam_utils import generate_gradcam

from fastai.vision.all import load_learner
from sklearn.model_selection import train_test_split
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_N = 50       # keep small for testing
MC_T = 20           # MC-Dropout forward passes
SAVE_ARTIFACTS = True
NUM_WORKERS = 0     # Windows = 0, Linux can increase

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

# ---------------- Paths ----------------
DATA_ROOT = Path("../COVID-19_Radiography_Dataset")
MODEL_PATH = Path("../models/mobilenetv3_small_100_cpu_fastai.pkl")
OUTPUT_DIR = Path("../outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16

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
logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")

# ---------------- Evaluation ----------------
test_files = list(test_df["path"].values)
test_dl = learn.dls.test_dl(test_files, bs=BATCH_SIZE, num_workers=NUM_WORKERS)

preds, _ = learn.get_preds(dl=test_dl)

true_labels = np.array([learn.dls.vocab.o2i[label] for label in test_df["label"].values])
pred_labels = preds.argmax(dim=1).cpu().numpy()

# Classification report
report = classification_report(true_labels, pred_labels, target_names=learn.dls.vocab)
logger.info("Classification Report (Test Set):\n" + report)

# AUROC + AUPRC
roc_auc = MulticlassAUROC(num_classes=len(learn.dls.vocab), average="macro")(preds, torch.tensor(true_labels)).item()
auprc = MulticlassAveragePrecision(num_classes=len(learn.dls.vocab), average="macro")(preds, torch.tensor(true_labels)).item()
logger.info(f"Test AUROC: {roc_auc:.4f}")
logger.info(f"Test AUPRC: {auprc:.4f}")

# ---------------- Calibration ----------------
logger.info("Running temperature scaling (calibration)...")
cal_model = ModelWithTemperature(learn.model).to(DEVICE)
temp = cal_model.set_temperature(learn.dls[1], device=DEVICE, max_iter=50)
logger.info(f"Found temperature = {temp:.3f}")

# compute calibrated probs
logger.info("Computing calibrated probabilities for test set...")
probs_list = []
for batch in test_dl:
    xb = batch[0] if isinstance(batch, (tuple, list)) else batch
    xb = xb.to(DEVICE)
    with torch.no_grad():
        logits = cal_model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    probs_list.append(probs)
probs_calibrated = np.vstack(probs_list)

# ECE + reliability diagram
ece = compute_ece(probs_calibrated, true_labels, n_bins=15)
rd_path = OUTPUT_DIR / "reliability_diagram.png"
plot_reliability_diagram(probs_calibrated, true_labels, rd_path, n_bins=15)
logger.info(f"ECE (test, after temperature scaling): {ece:.4f}")
logger.info(f"Reliability diagram saved to: {rd_path}")

# ---------------- MC-Dropout ----------------
logger.info(f"Running MC-Dropout (T={MC_T}) to estimate uncertainty...")
mc_preds = mc_dropout_predict(cal_model.model, test_dl, T=MC_T, device=DEVICE)
mc_mean = mc_preds.mean(axis=0)
mc_std = mc_preds.std(axis=0)

# Save per-sample artifacts
# optionally save per-sample artifacts (overlay + JSON)
if SAVE_ARTIFACTS:
    import json
    artifacts_dir = OUTPUT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # reset index so row positions match mc_mean
    test_df = test_df.reset_index(drop=True)

    # decide which positions to save
    if SAMPLE_N is None:
        positions = range(len(test_df))
    else:
        positions = list(test_df.sample(min(SAMPLE_N, len(test_df)), random_state=42).index)

    for rank, pos in enumerate(positions):
        img_path = test_df.loc[pos, "path"]
        # Grad-CAM overlay
        vis, pred_cls = generate_gradcam(learn, img_path, device=DEVICE)
        out_img = artifacts_dir / f"{Path(img_path).stem}_gradcam.png"
        plt.imsave(out_img, vis)

        # save JSON (probs + uncertainty)
        data = {
            "image": str(img_path),
            "predicted_class": int(pred_cls),
            "probs_calibrated": mc_mean[pos].tolist(),   # now matches!
            "uncertainty_std": mc_std[pos].tolist()
        }
        with open(artifacts_dir / f"{Path(img_path).stem}.json", "w") as f:
            json.dump(data, f, indent=2)

    logger.info(f"Saved {len(positions)} artifacts to {artifacts_dir}")

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

logger.info("Pipeline complete. Grad-CAM integration ready for next milestone.")
