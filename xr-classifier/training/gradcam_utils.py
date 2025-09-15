# gradcam_utils.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from fastai.vision.all import PILImage

def generate_gradcam(learn, img_path, target_layer=None):
    """
    Generate Grad-CAM heatmap for a given image and model.
    """
    model = learn.model
    model.eval()

    if target_layer is None:
        # pick last conv layer for MobileNetV3
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

    # hook for activations
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    # preprocess
    img = PILImage.create(img_path)
    dl = learn.dls.test_dl([img])
    inp = dl.one_batch()[0]

    # forward + backward
    out = model(inp)
    class_idx = out.argmax(dim=1).item()
    out[0, class_idx].backward()

    # grad-cam
    grads = gradients[0].mean(dim=(2,3), keepdim=True)
    cam = torch.relu((activations[0] * grads).sum(dim=1)).squeeze().cpu().numpy()

    # normalize
    cam = cv2.resize(cam, (inp.shape[2], inp.shape[3]))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # overlay
    img_np = np.array(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = 0.5 * heatmap + 0.5 * img_np

    handle_fwd.remove()
    handle_bwd.remove()

    return overlay.astype(np.uint8), class_idx
