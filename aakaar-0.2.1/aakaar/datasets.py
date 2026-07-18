"""
aakaar.datasets — ready-made Dataset implementations for common data
sources: CSV files and image folders.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from .data import Dataset


class CSVDataset(Dataset):
    """Loads a CSV into memory once, splits into feature columns and a
    target column (or columns)."""

    def __init__(self, path, target_col=-1, feature_cols=None, dtype=np.float32):
        df = pd.read_csv(path)

        if isinstance(target_col, str):
            target_col = df.columns.get_loc(target_col)

        if feature_cols is None:
            feature_cols = [i for i in range(df.shape[1]) if i != target_col]

        self.X = df.iloc[:, feature_cols].to_numpy(dtype=dtype)
        self.y = df.iloc[:, target_col].to_numpy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImageFolderDataset(Dataset):
    """ImageFolder: expects
        root/class_a/img1.jpg
        root/class_a/img2.jpg
        root/class_b/img3.jpg
    Each subdirectory of `root` becomes a class, alphabetically indexed.
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(valid_ext):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        return img, label