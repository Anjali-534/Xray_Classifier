import base64
import io
import random
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image, ImageOps
import numpy as np

from .utils.image_io import load_pil_rgb
from .gradcam import fake_heatmap_overlay

# In Milestone B/C we'll load a real PyTorch model. For Milestone A we stub.

@dataclass
class StubOutput:
    classes: List[str]
    probs: List[float]
    topk: List[dict]
    overlay_b64: str

DEFAULT_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Effusion", "Pneumonia", "Pneumothorax", "Fracture"
]

class Predictor:
    def __init__(self, classes: List[str] = None):
        self.classes = classes or DEFAULT_CLASSES

    def _encode_png_b64(self, pil_image: Image.Image) -> str:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def predict_stub(self, image_bytes: bytes) -> StubOutput:
        # Load, normalize to 512px longest side for nicer overlay
        pil = load_pil_rgb(image_bytes)
        pil = ImageOps.contain(pil, (1024, 1024))

        # Fake probabilities to exercise UI; seed random by image size for repeatability
        rnd = random.Random(len(image_bytes))
        probs = [round(max(0.01, min(0.99, rnd.random() * 0.9)), 2) for _ in self.classes]

        # Make a synthetic heatmap overlay for the top class
        top_idx = int(np.argmax(np.array(probs)))
        overlay = fake_heatmap_overlay(pil, seed=top_idx)

        overlay_b64 = self._encode_png_b64(overlay)
        topk = [{"class": self.classes[top_idx], "p": float(probs[top_idx])}]

        return StubOutput(classes=self.classes, probs=probs, topk=topk, overlay_b64=overlay_b64)
