# xr_classifier/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: List[float]
    uncertainty: Optional[List[float]] = None
    gradcam_path: Optional[str] = None
