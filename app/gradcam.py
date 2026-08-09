from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision.models import resnet18

from utils import preprocess_image, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

model.load_state_dict(
    torch.load("models/dermvision_resnet18.pth", map_location=device)
)

model = model.to(device)
model.eval()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor):
        image_tensor = image_tensor.to(device)

        output = self.model(image_tensor)
        pred_class = output.argmax(dim=1)

        self.model.zero_grad()
        output[0, pred_class].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], device=device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


gradcam = GradCAM(model, model.layer4)


def generate_gradcam(image):
    image_tensor = preprocess_image(image)

    heatmap = gradcam.generate(image_tensor)

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_np = np.array(image)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 0.7, heatmap, 0.3, 0)

    return Image.fromarray(overlay)