from PIL import Image
from torchvision import transforms
import torch

CLASS_NAMES = [
    "Actinic keratoses (AKIEC)",
    "Basal cell carcinoma (BCC)",
    "Benign keratosis-like lesions (BKL)",
    "Dermatofibroma (DF)",
    "Melanoma (MEL)",
    "Melanocytic nevus (NV)",
    "Vascular lesions (VASC)"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


def preprocess_image(image):

    if isinstance(image, str):
        image = Image.open(image)

    image = image.convert("RGB")
    image = IMAGE_TRANSFORM(image)
    image = image.unsqueeze(0)

    return image


def format_prediction(probabilities):

    predicted_index = torch.argmax(probabilities).item()
    confidence = probabilities[0][predicted_index].item() * 100

    class_probabilities = {
        CLASS_NAMES[i]: round(probabilities[0][i].item() * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    class_probabilities = dict(
        sorted(
            class_probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 2),
        "probabilities": class_probabilities
    }