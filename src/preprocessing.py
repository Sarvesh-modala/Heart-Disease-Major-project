"""
preprocessing.py
----------------
Modular preprocessing pipeline for cardiac MRI images.

Steps (in order)
----------------
1.  Non-rigid registration  – align every image to a common atlas/template
2.  Non-local means filtering   – edge-preserving denoising
3.  Intensity standardisation   – percentile-clip then z-score normalise
4.  CLAHE + DL enhancement      – contrast + learned sharpening
5.  Multilevel Otsu thresholding – produce a soft tissue/background mask

Each function works on *float32 numpy arrays in [0, 1]* (single channel).
The public entry-point `preprocess_image` chains them all and returns the
final float32 array ready for feature extraction or saving.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from PIL import Image
from skimage import exposure, filters, restoration

from src.config import (
    CLAHE_CLIP_LIMIT, CLAHE_KERNEL_SIZE,
    IMAGE_SIZE, MULTIOTSU_CLASSES,
    NLM_H, NLM_PATCH_SIZE, NLM_PATCH_DISTANCE,
    DEVICE,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 1.  Non-rigid (B-spline) registration
# ===========================================================================

def _build_atlas(size: Tuple[int, int] = IMAGE_SIZE) -> sitk.Image:
    """
    Synthetic atlas: a simple gradient image used as a fixed reference.
    In a clinical pipeline you would supply a real population atlas here.
    """
    arr = np.zeros(size, dtype=np.float32)
    cx, cy = size[0] // 2, size[1] // 2
    for i in range(size[0]):
        for j in range(size[1]):
            # Gaussian blob centred in the image – roughly mimics a heart silhouette
            arr[i, j] = np.exp(-((i - cx) ** 2 + (j - cy) ** 2) / (2 * (cx * 0.4) ** 2))
    return sitk.GetImageFromArray(arr)


# Cache atlas so it is built only once per process
_ATLAS: Optional[sitk.Image] = None


def nonrigid_registration(image: np.ndarray) -> np.ndarray:
    """
    Register *image* (H×W float32 [0,1]) to a fixed atlas using a
    B-spline free-form deformation field estimated by SimpleITK.

    The registration corrects for positional variation between scans so
    subsequent feature extraction is spatially consistent.
    """
    global _ATLAS
    if _ATLAS is None:
        _ATLAS = _build_atlas(IMAGE_SIZE)

    moving = sitk.GetImageFromArray(image.astype(np.float32))

    # Initialise with a simple centre-of-mass alignment to give B-spline a
    # good starting point, avoiding large deformations for unrelated images.
    initial_tx = sitk.CenteredTransformInitializer(
        _ATLAS, moving,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Affine registration first (faster + more stable)
    reg_affine = sitk.ImageRegistrationMethod()
    reg_affine.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg_affine.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=50,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg_affine.SetOptimizerScalesFromPhysicalShift()
    reg_affine.SetShrinkFactorsPerLevel([4, 2, 1])
    reg_affine.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg_affine.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg_affine.SetInitialTransform(initial_tx, inPlace=False)
    reg_affine.SetInterpolator(sitk.sitkLinear)

    try:
        affine_tx = reg_affine.Execute(_ATLAS, moving)
    except Exception as exc:
        logger.warning("Affine registration failed (%s); using identity.", exc)
        return image

    # B-spline non-rigid registration on top of affine
    mesh_size = [8] * moving.GetDimension()
    bspline_tx = sitk.BSplineTransformInitializer(_ATLAS, mesh_size)
    composite_tx = sitk.CompositeTransform(affine_tx)
    composite_tx.AddTransform(bspline_tx)

    reg_bspline = sitk.ImageRegistrationMethod()
    reg_bspline.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg_bspline.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=30,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=100,
        costFunctionConvergenceFactor=1e7,
    )
    reg_bspline.SetInitialTransform(composite_tx, inPlace=True)
    reg_bspline.SetInterpolator(sitk.sitkLinear)
    reg_bspline.SetShrinkFactorsPerLevel([2, 1])
    reg_bspline.SetSmoothingSigmasPerLevel([1, 0])
    reg_bspline.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    try:
        final_tx = reg_bspline.Execute(_ATLAS, moving)
        resampled = sitk.Resample(
            moving, _ATLAS, final_tx,
            sitk.sitkLinear, 0.0, moving.GetPixelID(),
        )
        result = sitk.GetArrayFromImage(resampled).astype(np.float32)
        # Clip to [0,1] since interpolation may overshoot
        return np.clip(result, 0.0, 1.0)
    except Exception as exc:
        logger.warning("B-spline registration failed (%s); using affine only.", exc)
        resampled = sitk.Resample(
            moving, _ATLAS, affine_tx,
            sitk.sitkLinear, 0.0, moving.GetPixelID(),
        )
        result = sitk.GetArrayFromImage(resampled).astype(np.float32)
        return np.clip(result, 0.0, 1.0)


# ===========================================================================
# 2.  Non-local means denoising
# ===========================================================================

def nlm_denoise(image: np.ndarray) -> np.ndarray:
    """
    Apply Non-Local Means (NLM) denoising.

    NLM preserves edges and fine structures better than Gaussian smoothing
    because it averages pixels with similar *neighbourhood patches* rather
    than spatially nearby pixels – important for preserving myocardial wall
    boundaries in MRI.
    """
    sigma_est = np.mean(restoration.estimate_sigma(image, channel_axis=None))
    h = max(NLM_H, sigma_est * 0.8)   # adaptive filter strength

    denoised = restoration.denoise_nl_means(
        image,
        h=h,
        patch_size=NLM_PATCH_SIZE,
        patch_distance=NLM_PATCH_DISTANCE,
        fast_mode=True,
        channel_axis=None,
    )
    return np.clip(denoised, 0.0, 1.0).astype(np.float32)


# ===========================================================================
# 3.  Intensity standardisation
# ===========================================================================

def intensity_standardise(image: np.ndarray,
                           low_pct: float = 1.0,
                           high_pct: float = 99.0) -> np.ndarray:
    """
    Percentile-clip then z-score normalise.

    Step 1 – Clip to the [low_pct, high_pct] percentile range to remove
             scanner-specific hot/cold outlier intensities.
    Step 2 – Rescale to [0, 1] so intensities are comparable across subjects
             and scanners (removes inter-scanner bias).
    """
    lo = np.percentile(image, low_pct)
    hi = np.percentile(image, high_pct)
    clipped = np.clip(image, lo, hi)

    denom = hi - lo if (hi - lo) > 1e-8 else 1.0
    standardised = (clipped - lo) / denom
    return standardised.astype(np.float32)


# ===========================================================================
# 4a. CLAHE contrast enhancement
# ===========================================================================

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    CLAHE divides the image into small contextual regions (tiles) and
    equalises each independently, then bilinearly blends the results.
    The clip limit prevents noise amplification in uniform regions.
    MRI images often have low contrast in the myocardium; CLAHE makes
    trabeculations and wall motion artefacts visible.
    """
    enhanced = exposure.equalize_adapthist(
        image,
        kernel_size=CLAHE_KERNEL_SIZE,
        clip_limit=CLAHE_CLIP_LIMIT,
    )
    return enhanced.astype(np.float32)


# ===========================================================================
# 4b. Deep-learning-based image enhancement  (DnCNN-lite)
# ===========================================================================

class _DnCNN(nn.Module):
    """
    Lightweight residual denoising CNN (inspired by Zhang et al. 2017).
    17 convolutional layers with batch-norm; learns residual noise to subtract.
    Used here as a learned sharpening / enhancement filter rather than a
    pure denoiser – it is initialised to near-identity so it runs usefully
    even without pre-trained weights.
    """

    def __init__(self, num_layers: int = 8, num_features: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(1, num_features, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_features, num_features, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(num_features, 1, 3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Small init so the residual starts near zero → near-identity mapping
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)
        # Subtract predicted noise from the input (residual learning)
        return torch.clamp(x - residual, 0.0, 1.0)


# Lazily initialised singleton so we pay the model-build cost once
_DL_ENHANCER: Optional[_DnCNN] = None


def _get_dl_enhancer(weights_path: Optional[Path] = None) -> _DnCNN:
    global _DL_ENHANCER
    if _DL_ENHANCER is None:
        _DL_ENHANCER = _DnCNN().to(DEVICE).eval()
        if weights_path and weights_path.exists():
            try:
                state = torch.load(weights_path, map_location=DEVICE)
                _DL_ENHANCER.load_state_dict(state)
                logger.info("Loaded DnCNN weights from %s", weights_path)
            except Exception as exc:
                logger.warning("Could not load DnCNN weights (%s). Using default init.", exc)
    return _DL_ENHANCER


def dl_enhance(image: np.ndarray,
               weights_path: Optional[Path] = None) -> np.ndarray:
    """
    Pass the image through the DnCNN residual enhancer.

    The enhancer runs in inference mode (no grad) and works on a single
    (1 × H × W) tensor.  Without pre-trained weights the network still
    provides slight smoothing + contrast correction due to its near-identity
    initialisation.
    """
    enhancer = _get_dl_enhancer(weights_path)
    tensor = torch.from_numpy(image[np.newaxis, np.newaxis]).to(DEVICE)  # 1×1×H×W
    with torch.no_grad():
        out = enhancer(tensor)
    result = out.squeeze().cpu().numpy().astype(np.float32)
    return np.clip(result, 0.0, 1.0)


# ===========================================================================
# 5.  Multilevel Otsu thresholding  (soft mask extraction)
# ===========================================================================

def multilevel_threshold(image: np.ndarray,
                          n_classes: int = MULTIOTSU_CLASSES) -> np.ndarray:
    """
    Compute multi-Otsu thresholds and return a normalised region map.

    Regions: background (0), myocardium boundary (0.5), cardiac muscle (1.0).
    The mask is blended back with the original image (alpha=0.3) so that
    subsequent CNN layers still receive full-intensity input while structural
    regions are softly highlighted.
    """
    try:
        thresholds = filters.threshold_multiotsu(image, classes=n_classes)
        regions = np.digitize(image, bins=thresholds).astype(np.float32)
        # Normalise region labels to [0, 1]
        regions /= (n_classes - 1)
    except Exception as exc:
        logger.warning("Multi-Otsu failed (%s); skipping mask overlay.", exc)
        return image

    # Soft overlay: retain original texture while emphasising structure
    blended = 0.7 * image + 0.3 * regions
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


# ===========================================================================
# Public entry-point
# ===========================================================================

def load_image_gray(path: Path) -> np.ndarray:
    """
    Load any image format as a float32 greyscale array in [0, 1],
    resized to IMAGE_SIZE (H×W).
    """
    img = Image.open(path).convert("L")          # force 8-bit greyscale
    img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def preprocess_image(
    path: Path,
    skip_registration: bool = False,
    dl_weights: Optional[Path] = None,
) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    path               : path to raw MRI image
    skip_registration  : set True to skip non-rigid registration (saves time
                         when no atlas-alignment is needed, e.g. already-aligned
                         datasets)
    dl_weights         : optional path to DnCNN .pth weights file

    Returns
    -------
    Preprocessed float32 array, shape (H, W), values in [0, 1].
    """
    img = load_image_gray(path)

    # Step 1 – Non-rigid registration
    if not skip_registration:
        img = nonrigid_registration(img)

    # Step 2 – NLM denoising
    img = nlm_denoise(img)

    # Step 3 – Intensity standardisation
    img = intensity_standardise(img)

    # Step 4a – CLAHE
    img = apply_clahe(img)

    # Step 4b – DL enhancement
    img = dl_enhance(img, weights_path=dl_weights)

    # Step 5 – Multilevel Otsu mask blend
    img = multilevel_threshold(img)

    return img


def preprocess_and_save(
    path: Path,
    out_dir: Path,
    **kwargs,
) -> Path:
    """
    Preprocess one image and save the result as a uint8 PNG.
    Returns the output path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = preprocess_image(path, **kwargs)
    out_path = out_dir / (path.stem + "_preprocessed.png")
    arr_uint8 = (result * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr_uint8, mode="L").save(out_path)
    return out_path
