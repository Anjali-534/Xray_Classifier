from PIL import Image
import numpy as np

def fake_heatmap_overlay(pil_img: Image.Image, seed: int = 0) -> Image.Image:
    """Create a translucent colored blob overlay to prove the UI path."""
    w, h = pil_img.size
    rng = np.random.default_rng(seed)
    heat = rng.normal(loc=0.5, scale=0.25, size=(h, w))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    # colormap to RGBA (simple red channel)
    alpha = (heat * 180).astype(np.uint8)
    red = (heat * 255).astype(np.uint8)
    overlay = Image.fromarray(np.dstack([red, np.zeros_like(red), np.zeros_like(red), alpha]), mode="RGBA")
    out = pil_img.convert("RGBA").copy()
    out.alpha_composite(overlay)
    return out.convert("RGBA")
