from PIL import Image
import io
def load_pil_rgb(image_bytes: bytes) -> Image.Image:
    pil= Image.open(io.BytesIO(image_bytes))
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil