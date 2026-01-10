import os
import torch
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image


dataset_path="EuroSAT_RGB"
class_names = sorted(
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith(".")
)

class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
print(class_to_idx)

class RGB_Dataset(Dataset):
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

class MS_Dataset(Dataset):
    def __init__(self, split_file, band_indices, transform=None):
        self.band_indices = band_indices
        self.transform = transform
        self.samples = []

        with open(split_file) as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Load image: (H, W, C)
        img = imread(path)

        # Select bands and reorder to (C, H, W)
        bands = img[:, :, self.band_indices]
        bands = np.transpose(bands, (2, 0, 1))

        image = bands.astype(np.float32) / 65535.0
        image = torch.from_numpy(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class MS_Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return F.interpolate(x.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)
