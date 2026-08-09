from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18

from utils import preprocess_image, format_prediction, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

model_path = Path(__file__).resolve().parent.parent / "models" / "dermvision_resnet18.pth"

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()


def predict_image(image):
    
    image = preprocess_image(image)
    image = image.to(device)

    with torch.inference_mode():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

    result = format_prediction(probabilities)

    return result