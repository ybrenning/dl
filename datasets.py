import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image


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

        with rasterio.open(path) as src:
            bands = src.read([b + 1 for b in self.band_indices])
            # (6, H, W)

        image = bands.astype(np.float32) / 65535.
        image = torch.from_numpy(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class MS_Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return F.interpolate(x.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False).squeeze(0)


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

        with rasterio.open(path) as src:
            bands = src.read([b + 1 for b in self.band_indices])
            # (6, H, W)

        image = bands.astype(np.float32) / 65535.
        image = torch.from_numpy(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
