# X-ray Classifier (Demo)

Upload a chest X-ray -> multi-label predictions with a heatmap overlay.

> Research demo; not for clinical use.

## Quickstart
```bash
# API
uvicorn serving.app:app --reload --port 8000
# Web
cd webapp && npm install && npm run dev
