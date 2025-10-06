# X-ray Classifier (Demo)

Upload a chest X-ray -> multi-label predictions with a heatmap overlay.

> Research demo; not for clinical use.

## Quickstart
```bash
# API
uvicorn serving.app:app --reload --port 8000
http://127.0.0.1:8000/docs

# Web
cd webapp && npm install && npm run dev
# to train model
run python train.py or python train_fastai_timm.py
run evaluate.py