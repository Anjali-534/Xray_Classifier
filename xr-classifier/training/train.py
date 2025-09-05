import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset_covid import CovidXrayDataset, CLASSES
from transforms import get_train_transforms, get_valid_transforms

# ---------------- CONFIG ----------------
DATA_ROOT = "../COVID-19_Radiography_Dataset"

CHECKPOINT_DIR = "../models"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", DEVICE)
print("🔎 Checking dataset folders...")
for c in CLASSES:
    folder = os.path.join(DATA_ROOT, c,"images") 
    if not os.path.isdir(folder):
        print(f"❌ Missing folder: {folder}")
    else:
        print(f"{c} -> {len([f for f in os.listdir(folder) if f.lower().endswith('.png')])} images")


# ---------------- DATASET ----------------
all_samples = []
for c in CLASSES:
    img_dir = os.path.join(DATA_ROOT, c,"images")  # <-- dive into images/
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(DATA_ROOT, c)

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


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ---------------- MODEL ----------------
def get_model(num_classes=len(CLASSES)):
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

model = get_model().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# ---------------- TRAINING LOOP ----------------
best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_acc = correct / total

    # validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: Train Acc {train_acc:.3f}, Val Acc {val_acc:.3f}, Loss {running_loss/len(train_loader):.3f}")
    scheduler.step(val_acc)

    # save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_densenet121.pth"))
        print("✅ Saved best model")

# ---------------- TEST EVALUATION ----------------
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_densenet121.pth")))
model.eval()

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()
