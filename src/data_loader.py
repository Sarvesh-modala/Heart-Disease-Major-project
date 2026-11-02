"""
data_loader.py
--------------
Discovers images on disk, enforces the four-class taxonomy, applies
stratified train/test splitting with optional oversampling for class
balance, and returns ready-to-use PyTorch DataLoaders.

Expected on-disk layout
-----------------------
data/heart_mri/
    cardiovascular_disease/  *.png | *.jpg | *.jpeg | *.bmp | *.tif
    cardiomyopathy/
    congenital_heart_disease/
    normal/
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from src.config import (
    DATASET_ROOT, CLASSES, CLASS_TO_IDX, IMAGE_SIZE,
    TRAIN_SPLIT, RANDOM_SEED, BATCH_SIZE, NUM_CLASSES,
)

logger = logging.getLogger(__name__)

# Supported image extensions
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Disk scanning
# ---------------------------------------------------------------------------

def discover_dataset(root: Path = DATASET_ROOT) -> Dict[str, List[Path]]:
    """
    Walk *root* and return a mapping  {class_name: [image_paths]}.
    Missing class folders emit a warning; completely empty roots raise.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Please download the Kaggle dataset and place it under data/heart_mri/ "
            "(see README.md for instructions)."
        )

    dataset: Dict[str, List[Path]] = {}
    for cls in CLASSES:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            logger.warning("Class folder missing: %s  — skipping.", cls_dir)
            continue
        images = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in _IMG_EXTS
        )
        if not images:
            logger.warning("No images found in %s — skipping.", cls_dir)
            continue
        dataset[cls] = images
        logger.info("Class %-30s → %d images", cls, len(images))

    if not dataset:
        raise RuntimeError(
            f"No valid class folders found under {root}.\n"
            f"Expected sub-folders: {CLASSES}"
        )
    return dataset


def split_dataset(
    dataset: Dict[str, List[Path]],
    train_ratio: float = TRAIN_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Stratified split → (train_samples, test_samples).
    Each sample is a (path, class_index) tuple.
    """
    rng = random.Random(seed)
    train_samples: List[Tuple[Path, int]] = []
    test_samples:  List[Tuple[Path, int]] = []

    for cls, paths in dataset.items():
        idx = CLASS_TO_IDX[cls]
        shuffled = paths[:]
        rng.shuffle(shuffled)
        split_at = max(1, int(len(shuffled) * train_ratio))
        train_samples.extend((p, idx) for p in shuffled[:split_at])
        test_samples.extend( (p, idx) for p in shuffled[split_at:])

    logger.info("Train samples: %d  |  Test samples: %d",
                len(train_samples), len(test_samples))
    return train_samples, test_samples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class HeartMRIDataset(Dataset):
    """
    Loads grayscale MRI images, converts to 3-channel (ResNet expects RGB),
    and applies the supplied transform.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")   # always 3-channel
        except Exception as exc:
            logger.error("Cannot open %s: %s", path, exc)
            # Return a black placeholder so training doesn't crash
            img = Image.new("RGB", IMAGE_SIZE, 0)

        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(train: bool = True) -> transforms.Compose:
    """
    Training: random flips + colour jitter for augmentation.
    Evaluation: only resize + normalise (ImageNet stats for ResNet).
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Weighted sampler for class balance
# ---------------------------------------------------------------------------

def make_weighted_sampler(samples: List[Tuple[Path, int]]) -> WeightedRandomSampler:
    """
    Over-sample minority classes so every batch is approximately balanced.
    Weight per sample = 1 / class_frequency.
    """
    labels = [lbl for _, lbl in samples]
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.where(class_counts == 0, 1, class_counts)   # avoid /0
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[lbl] for lbl in labels], dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    root: Path = DATASET_ROOT,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Full pipeline: discover → split → wrap in DataLoaders.

    Returns
    -------
    train_loader, test_loader, train_samples, test_samples
    """
    dataset = discover_dataset(root)
    train_samples, test_samples = split_dataset(dataset)

    train_ds = HeartMRIDataset(train_samples, transform=build_transforms(train=True))
    test_ds  = HeartMRIDataset(test_samples,  transform=build_transforms(train=False))

    sampler = make_weighted_sampler(train_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,            # weighted → no shuffle needed
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_samples, test_samples
