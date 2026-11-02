"""
feature_extraction.py
---------------------
Two complementary feature extraction strategies:

A. Classical — Histogram of Oriented Gradients (HOG)
   Captures local shape and edge patterns; interpretable and lightweight.

B. Deep — ResNet-50 global-average-pool features
   2048-dimensional embedding learned from ImageNet + fine-tuned on our
   cardiac data; captures high-level semantic content.

Both feature sets are saved as .npy files so they can be reused without
re-running the full pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import hog
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from src.config import (
    DEVICE, FEATURES_DIR, HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, IMAGE_SIZE,
    RESNET_FEATURE_DIM,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# A.  HOG feature extraction
# ===========================================================================

def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Compute HOG descriptor for a single float32 greyscale image (H×W).

    HOG divides the image into small cells and computes a histogram of
    gradient orientations in each cell.  The concatenated normalised
    histograms form the descriptor.  Orientation captures cardiac texture
    and boundary information robustly to illumination changes.

    Returns
    -------
    1-D float32 feature vector.
    """
    fd = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=False,
        channel_axis=None,     # single-channel input
    )
    return fd.astype(np.float32)


def build_hog_feature_matrix(
    samples: List[Tuple[Path, int]],
    preprocessed_dir: Optional[Path] = None,
    save_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract HOG features for every sample and optionally save results.

    Parameters
    ----------
    samples          : list of (image_path, label_index) tuples
    preprocessed_dir : if given, look for preprocessed PNGs here first;
                       otherwise load raw images on-the-fly (slower)
    save_path        : where to save (X.npy, y.npy) pair

    Returns
    -------
    X : (N, D) float32 feature matrix
    y : (N,)   int label array
    """
    from src.preprocessing import load_image_gray, preprocess_image

    features, labels = [], []

    for path, label in tqdm(samples, desc="HOG extraction"):
        # Prefer a pre-processed image if it exists
        if preprocessed_dir is not None:
            pp_path = preprocessed_dir / path.parent.name / (path.stem + "_preprocessed.png")
            img = load_image_gray(pp_path) if pp_path.exists() else preprocess_image(path)
        else:
            img = preprocess_image(path)

        features.append(extract_hog_features(img))
        labels.append(label)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.int64)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path) + "_X.npy", X)
        np.save(str(save_path) + "_y.npy", y)
        logger.info("HOG features saved → %s_X/y.npy  shape=%s", save_path, X.shape)

    return X, y


# ===========================================================================
# B.  Deep feature extraction with ResNet-50
# ===========================================================================

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-50 with the classification head removed.
    Output: 2048-dim global-average-pooled feature vector per image.
    Weights: ImageNet-pretrained (torchvision default).
    """

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Drop the final fully-connected layer; keep everything up to avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        if weights_path and weights_path.exists():
            try:
                state = torch.load(weights_path, map_location=DEVICE)
                # Allow partial loading (fine-tuned classifier weights not needed here)
                missing, unexpected = self.load_state_dict(state, strict=False)
                if missing:
                    logger.debug("Missing keys in feature extractor state: %s", missing[:5])
                logger.info("Loaded ResNet feature extractor from %s", weights_path)
            except Exception as exc:
                logger.warning("Could not load weights (%s). Using ImageNet init.", exc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)  →  out: (B, 2048)
        return self.features(x).flatten(1)


@torch.no_grad()
def extract_deep_features(
    loader: DataLoader,
    weights_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run all images through the ResNet-50 feature extractor.

    Parameters
    ----------
    loader       : DataLoader whose dataset yields (tensor, label)
    weights_path : optional path to a .pth checkpoint with resnet weights
    save_path    : optional base path; saves <save_path>_X.npy and _y.npy

    Returns
    -------
    X : (N, 2048) float32 deep feature matrix
    y : (N,)      int label array
    """
    extractor = ResNetFeatureExtractor(weights_path).to(DEVICE).eval()

    all_feats, all_labels = [], []
    for images, labels in tqdm(loader, desc="Deep feature extraction"):
        images = images.to(DEVICE)
        feats  = extractor(images)                # (B, 2048)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())

    X = np.concatenate(all_feats,  axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path) + "_X.npy", X)
        np.save(str(save_path) + "_y.npy", y)
        logger.info("Deep features saved → %s_X/y.npy  shape=%s", save_path, X.shape)

    return X, y


# ===========================================================================
# Utility: load saved features
# ===========================================================================

def load_features(base_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load previously saved X_*.npy and y_*.npy pairs."""
    X = np.load(str(base_path) + "_X.npy")
    y = np.load(str(base_path) + "_y.npy")
    logger.info("Loaded features from %s  shape=%s", base_path, X.shape)
    return X, y
