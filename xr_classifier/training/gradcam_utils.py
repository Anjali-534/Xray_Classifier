# gradcam_utils.py
import numpy as np
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from fastai.vision.all import PILImage, imagenet_stats

def _select_conv_layer(model):
    # Attempt to find a last conv layer robustly (works for MobileNet, TIMM models)
    if hasattr(model, "features"):
        # search backwards for a Conv2d-containing module
        for m in reversed(list(model.features)):
            # check children types
            for c in m.children() if hasattr(m, "children") else []:
                if c.__class__.__name__.lower().startswith("conv"):
                    return m
        return model.features[-1]
    # fallback: find last child module that contains conv
    children = list(model.children())
    for m in reversed(children):
        for c in m.children() if hasattr(m, "children") else []:
            if c.__class__.__name__.lower().startswith("conv"):
                return m
    return children[-2] if len(children) > 1 else children[-1]

def generate_gradcam(learn, img_path, device="cpu"):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model = learn.model
    model.eval()

    # MobileNetV3 feature maps before pooling
    target_layers = [model.conv_stem] if hasattr(model, "conv_stem") else [list(model.children())[-2]]

    # Load and preprocess
    img = PILImage.create(img_path)
    input_tensor = learn.dls.test_dl([img_path]).one_batch()[0].to(device)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        preds = model(input_tensor)
        pred = preds.argmax(dim=1).item()
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred)])
        grayscale_cam = grayscale_cam[0, :]

    # Normalize input image to overlay
    rgb_img = np.array(img.resize((input_tensor.shape[-1], input_tensor.shape[-2]))) / 255.0
    visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)

    return visualization, pred
