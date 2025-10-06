# xr_classifier/inference.py
import torch
import numpy as np
from pathlib import Path
from fastai.vision.all import load_learner, PILImage
from  training.calibration_utils import ModelWithTemperature
from .training.mc_dropout_utils import mc_dropout_predict
from .training.gradcam_utils import generate_gradcam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/mobilenetv3_small_100_cpu_fastai.pkl")
OUTPUT_DIR = Path("outputs/api_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model once
learn = load_learner(MODEL_PATH)
learn.model.eval().to(DEVICE)
cal_model = ModelWithTemperature(learn.model).to(DEVICE)

def predict_image(img_path: str, mc_dropout: bool = True, gradcam: bool = True):
    """Run full inference pipeline on one image."""
    img = PILImage.create(img_path)
    dl = learn.dls.test_dl([img])
    preds, _ = learn.get_preds(dl=dl)
    
    probs = preds[0].cpu().numpy()
    pred_class_idx = probs.argmax()
    pred_class = learn.dls.vocab[pred_class_idx]

    response = {
        "predicted_class": pred_class,
        "probabilities": probs.tolist()
    }

    # MC-Dropout uncertainty
    if mc_dropout:
        mc_preds = mc_dropout_predict(cal_model.model, dl, T=20, device=DEVICE)
        mc_std = mc_preds.std(axis=0)[0]  # std for this sample
        response["uncertainty"] = mc_std.tolist()

    # Grad-CAM visualization
    if gradcam:
        vis, _ = generate_gradcam(learn, img_path, device=DEVICE)
        out_path = OUTPUT_DIR / f"{Path(img_path).stem}_gradcam.png"
        import matplotlib.pyplot as plt
        plt.imsave(out_path, vis)
        response["gradcam_path"] = str(out_path)

    return response
