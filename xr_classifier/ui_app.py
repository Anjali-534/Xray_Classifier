import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="X-Ray Classifier", layout="centered")

st.title("🩻 Chest X-Ray Classifier with Grad-CAM + Uncertainty")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send to backend
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/predict/", files=files)

    if response.status_code == 200:
        result = response.json()
        st.subheader(f"Prediction: {result['predicted_class']}")
        st.write("Class probabilities:", result["probs"])
        st.write("Uncertainty (std):", result["uncertainty_std"])

        # Grad-CAM preview
        gradcam_response = requests.post(f"{API_URL}/gradcam/", files=files)
        if gradcam_response.status_code == 200:
            gradcam_img = Image.open(io.BytesIO(gradcam_response.content))
            st.image(gradcam_img, caption="Grad-CAM Overlay", use_column_width=True)
    else:
        st.error("Error: Could not get prediction from backend")
