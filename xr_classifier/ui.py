# ui.py
import gradio as gr
import numpy as np
from inference import predict_image

def inference_interface(image):
    result = predict_image(image, return_visuals=True)

    pred_class = result["predicted_class"]
    probs = {str(i): float(p) for i, p in enumerate(result["calibrated_probs"])}
    overlay = result["gradcam_overlay"]

    return pred_class, probs, overlay

demo = gr.Interface(
    fn=inference_interface,
    inputs=gr.Image(type="filepath", label="Upload X-ray"),
    outputs=[
        gr.Label(label="Predicted Class"),
        gr.Label(label="Calibrated Probabilities"),
        gr.Image(type="numpy", label="Grad-CAM Overlay"),
    ],
    title="Chest X-ray Classifier",
    description="Upload an X-ray to get diagnosis + uncertainty + Grad-CAM visualization"
)

if __name__ == "__main__":
    demo.launch()
