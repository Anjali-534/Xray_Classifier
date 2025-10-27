from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from .inference import Predictor, StubOutput

APP_VERSION = "d121-chexpert-2025-08-01"  # placeholder

app = FastAPI(title="XR-Classifier API ", version=APP_VERSION)

# CORS: adjust in docker/.env later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor()  # loads model lazily

class PredictResponse(BaseModel):
    classes: List[str]
    probs: List[float]
    probs_cal: Optional[List[float]] = None
    uncertainty_std: Optional[List[float]] = None
    topk: List[dict]
    overlay_png_base64: str
    model_version: str
    disclaimer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"model_version": APP_VERSION}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # ✅ Validate file type
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only PNG/JPEG images are supported.")

    # ✅ Validate size
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 20MB).")

    # 🧠 Stub prediction (until real inference is integrated)
    pred_class = "Normal"
    confidence = 0.94
    overlay_base64 = ""  # You can later embed Grad-CAM overlay here

    # ✅ Return response that matches PredictResponse model
    return {
        "classes": [pred_class],
        "probs": [confidence],
        "probs_cal": [confidence],
        "uncertainty_std": [0.05],
        "topk": [{"class": pred_class, "prob": confidence}],
        "overlay_png_base64": overlay_base64,
        "model_version": APP_VERSION,
        "disclaimer": "Research demo; not for clinical use."
    }

