from pathlib import Path
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader


def train_step(model, dataloader, loss_fn, optimizer, device, epoch, epochs):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    total_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
            print(f"Epoch {epoch}/{epochs} | Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.4f}")

    train_loss /= len(dataloader)
    train_acc = (correct / total) * 100

    return train_loss, train_acc


def val_step(model, dataloader, loss_fn, device):
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(dataloader)
    val_acc = (correct / total) * 100

    return val_loss, val_acc


def test_step(model, dataloader, loss_fn, device):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(dataloader)
    test_acc = (correct / total) * 100

    return test_loss, test_acc


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_path = Path("data/processed/train")
    val_path = Path("data/processed/val")
    test_path = Path("data/processed/test")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10, fill=(230, 190, 180)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.5
    )

    epochs = 20
    best_val_acc = 0

    print("Starting training...")

    for epoch in range(epochs):
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_step(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            epoch + 1,
            epochs
        )

        val_loss, val_acc = val_step(
            model,
            val_loader,
            loss_fn,
            device
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/dermvision_resnet18.pth")

        print(
            f"Epoch {epoch + 1}/{epochs} Completed | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

    print("\nLoading best validation model...")
    model.load_state_dict(torch.load("models/dermvision_resnet18.pth", map_location=device))

    print("Evaluating on test set...")

    test_loss, test_acc = test_step(
        model,
        test_loader,
        loss_fn,
        device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()