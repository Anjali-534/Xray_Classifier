# inference_and_gradcam.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
from fastai.vision.all import load_learner
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# ---------- config ----------
MODEL_PATH = Path("../models/tf_efficientnet_b3_ns_fastai.pkl")  # set to exported model
IMAGE_PATH = Path("path/to/sample_image.png")                    # set to your X-ray path
OUT_PATH = Path("../outputs")
OUT_PATH.mkdir(parents=True, exist_ok=True)

# ---------- load learner ----------
learn = load_learner(MODEL_PATH)
model = learn.model.eval()

# ---------- predict ----------
img_pil = Image.open(IMAGE_PATH).convert("RGB")
pred,pred_idx,probs = learn.predict(IMAGE_PATH)
print("Prediction:", pred, "Probabilities:", probs)

# ---------- Grad-CAM ----------
# helper: find last conv layer (robust)
def find_last_conv(module):
    convs = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    if len(convs)==0:
        raise ValueError("No Conv2d found in model")
    return convs[-1]

target_layer = find_last_conv(model)

use_cuda = torch.cuda.is_available()
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)

# Preprocess image for model: fastai already uses imagenet_stats normalization.
# Build a numpy float image in range 0..1
img_arr = np.array(img_pil.resize((224,224))) / 255.0
input_tensor = preprocess_image(img_arr, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # returns numpy
input_tensor = torch.from_numpy(input_tensor).to(next(model.parameters()).device)

# Grad-CAM expects a batch
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(int(pred_idx))])
grayscale_cam = grayscale_cam[0, :]  # single image

# overlay
visualization = show_cam_on_image(img_arr, grayscale_cam, use_rgb=True)
out_file = OUT_PATH / "gradcam_overlay.png"
Image.fromarray(visualization).save(out_file)
print("Saved Grad-CAM overlay to:", out_file)
