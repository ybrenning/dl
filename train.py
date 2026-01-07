import torch
import torch.nn as nn
import torchvision.models as models
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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


def main():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.RandomCrop(224),
    #         #further augmentations here or after the totensor
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #     'val': transforms.Compose([
    #         transforms.Resize(224),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ]),
    #     }

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
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )


    num_classes = 10

    print("Using device:", device)
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

    for epoch in range(5):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion
        )

        print(
            f"Epoch {epoch+1}: "
            f"train_acc={train_acc:.3f}, "
            f"val_acc={val_acc:.3f}"
        )


if __name__ == "__main__":
    main()
