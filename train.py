import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.models as models
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

seed = 3732848
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if torch.backends.mps.is_available():
    torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 10

class EuroSATDataset(Dataset):
    def __init__(self, split_file, transform=None):
        self.transform = transform

        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                rel_path, label = line.strip().split()
                self.samples.append((rel_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


@torch.no_grad()
def per_class_accuracy(model, loader, num_classes):
    model.eval()
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for c in range(num_classes):
            mask = (labels == c)
            correct_per_class[c] += (preds[mask] == c).sum().item()
            total_per_class[c] += mask.sum().item()

    per_class_acc = correct_per_class / total_per_class
    return per_class_acc


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=total_loss/total,
            acc=correct/total
        )

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_and_evaluate(
    train_transform,
    transform_name,
    num_classes=10,
    max_epochs=10,
    patience=2,
    min_delta=1e-4,
):

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = EuroSATDataset(
        split_file="splits/train.txt",
        transform=train_transform
    )

    val_dataset = EuroSATDataset(
        split_file="splits/val.txt",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
    )

    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    all_per_class_acc = []

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion
        )

        per_class_acc = per_class_accuracy(model, val_loader, num_classes)
        all_per_class_acc.append(per_class_acc)

        print(
            f"[{transform_name}] Epoch {epoch+1}: "
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                f"eurosat_{transform_name}.pth"
            )
            print(f"[{transform_name}] Saved new best model!")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[{transform_name}] Early stopping "
                    f"(best_val_loss={best_val_loss:.4f})"
                )
                break

    accs = np.array(all_per_class_acc)
    np.save(
        f"results/per_class_acc_{transform_name}.npy",
        accs
    )

    return accs


def main():
    train_transform_simple = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    train_transform_strong = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

    simple_per_class_acc = train_and_evaluate(
        train_transform_simple,
        transform_name="simple"
    )

    strong_per_class_acc = train_and_evaluate(
        train_transform_strong,
        transform_name="strong"
    )


if __name__ == "__main__":
    main()
