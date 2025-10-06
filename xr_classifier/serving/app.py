from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from .inference import Predictor, StubOutput

APP_VERSION = "d121-chexpert-2025-08-01"  # placeholder

app = FastAPI(title="X-ray ", version=APP_VERSION)

# CORS: adjust in docker/.env later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Only PNG/JPEG images are supported.")
    data = await file.read()
    if len(data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 20MB).")

    # For Milestone A: return a stub that proves plumbing works.
    out: StubOutput = predictor.predict_stub(data)

    return PredictResponse(
        classes=out.classes,
        probs=out.probs,
        probs_cal=out.probs,  # same for stub
        uncertainty_std=[0.05 for _ in out.probs],
        topk=out.topk,
        overlay_png_base64=out.overlay_b64,
        model_version=APP_VERSION,
        disclaimer="Research demo; not for clinical use.",
    )
