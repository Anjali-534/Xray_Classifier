# evaluate.py
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from fastai.vision.all import *
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
import cv2
from gradcam_utils import generate_gradcam
from calibration_utils import ModelWithTemperature
from mc_dropout_utils import mc_dropout_predict
# 🔥 our new utilities
from gradcam_utils import generate_gradcam
from calibration_utils import ModelWithTemperature
from mc_dropout_utils import mc_dropout_predict

# ---------------- CONFIG ----------------
DATA_ROOT = Path("../COVID-19_Radiography_Dataset")
MODEL_PATH = Path("../models/mobilenetv3_small_100_cpu_fastai.pkl")
BATCH_SIZE = 16
NUM_WORKERS = 0

# ---------------- Load Dataset ----------------
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

df = collect_dataset(DATA_ROOT)
_, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)

print("Test size:", len(test_df))

# Define test_files as a list of image paths from test_df
test_files = test_df["path"].tolist()

# ---------------- Load Model ----------------
learn = load_learner(MODEL_PATH)
print(f"✅ Model loaded from: {MODEL_PATH}")

n_classes = len(learn.dls.vocab)

# ---------------- Test Set Evaluation ----------------
# ---------------- Evaluate ----------------
print("\n📊 Evaluation on test set...")

test_dl = learn.dls.test_dl(test_files, bs=BATCH_SIZE, num_workers=NUM_WORKERS)

# Force return_y=True to get labels
preds, targs = learn.get_preds(dl=test_dl, with_decoded=False, act=None, inner=True)

pred_labels = preds.argmax(dim=1).cpu().numpy()
true_labels = np.array(targs.cpu()) if targs is not None else np.array([f.label for f in test_files])

# sklearn classification report
print(classification_report(true_labels, pred_labels, target_names=learn.dls.vocab))

# AUROC + AUPRC
roc_auc = MulticlassAUROC(num_classes=n_classes, average="macro")(preds, targs)
auprc   = MulticlassAveragePrecision(num_classes=n_classes, average="macro")(preds, targs)

print(f"Test AUROC: {roc_auc:.4f}")
print(f"Test AUPRC: {auprc:.4f}")


# ---------------- Grad-CAM Example ----------------
print("\n🔥 Generating Grad-CAM for one test image...")
sample_img = test_df.iloc[0]["path"]
heatmap, class_idx = generate_gradcam(learn, sample_img)
plt.imshow(heatmap)
plt.title(f"Grad-CAM: {learn.dls.vocab[class_idx]}")
plt.axis("off")
plt.show()

# ---------------- Temperature Scaling ----------------
print("\n🌡️ Running temperature scaling on validation set...")
valid_dl = learn.dls.valid
model_temp = ModelWithTemperature(learn.model)
model_temp.set_temperature(valid_dl)

# Example calibrated inference
logits = model_temp(torch.from_numpy(np.expand_dims(preds[0].numpy(), axis=0)))
calibrated_probs = torch.softmax(logits, dim=1)
print("Calibrated probs example:", calibrated_probs.detach().numpy())

# ---------------- MC-Dropout ----------------
print("\n🎲 Running MC-Dropout inference on 1 batch...")
mc_preds = mc_dropout_predict(learn.model, test_dl, T=10)  # [T, N, C]
mean_preds = mc_preds.mean(axis=0)
std_preds = mc_preds.std(axis=0)

print("Uncertainty example (first sample):")
for i, cls in enumerate(learn.dls.vocab):
    print(f"  {cls}: {mean_preds[0,i]:.3f} ± {std_preds[0,i]:.3f}")


# ---------------- Artifact saving ----------------
ARTIFACT_DIR = Path("../artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📂 Saving artifacts to {ARTIFACT_DIR}")

# 1) Wrap model with temperature scaling (using validation set)
print("🔥 Calibrating model with temperature scaling...")
valid_dl = learn.dls.valid
cal_model = ModelWithTemperature(learn.model).set_temperature(valid_dl)

# 2) Run MC-Dropout on test set
print("🎲 Running MC-Dropout (T=20)...")
test_dl = learn.dls.test_dl(test_files, bs=BATCH_SIZE, num_workers=NUM_WORKERS)
mc_preds = mc_dropout_predict(cal_model, test_dl, T=20)  # shape [T, N, C]

# mean & std across T
mc_mean = mc_preds.mean(axis=0)  # [N, C]
mc_std  = mc_preds.std(axis=0)   # [N, C]

# 3) Save per-image results
for idx, img_path in enumerate(test_files):
    # --- Grad-CAM ---
    overlay, class_idx = generate_gradcam(learn, img_path)

    # save heatmap
    out_path = ARTIFACT_DIR / f"sample_{idx}_gradcam.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # --- Save JSON metadata ---
    result = {
        "image": str(img_path),
        "pred_class": dls.vocab[class_idx],
        "probs_calibrated": mc_mean[idx].tolist(),
        "uncertainty_std": mc_std[idx].tolist(),
    }
    with open(ARTIFACT_DIR / f"sample_{idx}.json", "w") as f:
        json.dump(result, f, indent=2)

print(f"✅ Saved {len(test_files)} Grad-CAM overlays and metadata to {ARTIFACT_DIR}")
