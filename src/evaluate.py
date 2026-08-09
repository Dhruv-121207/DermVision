from pathlib import Path
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_path = Path("data/processed/test")
model_path = Path("models/dermvision_resnet18.pth")

results_path = Path("results")
results_path.mkdir(exist_ok=True)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(test_dataset.classes))

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

all_labels = []
all_preds = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(
    all_labels,
    all_preds,
    target_names=test_dataset.classes,
    digits=4
)

print(report)

with open(results_path / "classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_dataset.classes,
    yticklabels=test_dataset.classes
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

plt.savefig(results_path / "confusion_matrix.png", dpi=300)
plt.show()