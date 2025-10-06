# xr_classifier/app.py
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
from inference import predict_image
from .schemas import PredictionResponse
from inference import predict_image, generate_gradcam

app = FastAPI(title="X-Ray Classifier API")

# Mount static directory for Grad-CAM previews
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app = FastAPI(title="X-Ray Classifier API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
def root():
    return {"message": "X-Ray Classifier API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    upload_path = STATIC_DIR / file.filename
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run model inference
    result = predict_image(str(upload_path))

    # Generate Grad-CAM overlay
    gradcam_img, pred_class = generate_gradcam(str(upload_path))
    gradcam_path = STATIC_DIR / f"gradcam_{file.filename}.png"
    gradcam_img.save(gradcam_path)  # assuming PIL.Image

    # Return JSON with direct Grad-CAM URL
    return JSONResponse({
        "prediction": result,
        "gradcam_url": f"/static/{gradcam_path.name}"
    })

