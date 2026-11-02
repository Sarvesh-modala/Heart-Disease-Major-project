"""
Central configuration for the Heart MRI Classification project.
All tunable constants live here; no magic numbers in other modules.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("data/heart_mri")          # Place Kaggle data here (see README)
OUTPUT_DIR   = Path("outputs")
MODEL_DIR    = OUTPUT_DIR / "models"
PLOT_DIR     = OUTPUT_DIR / "plots"
FEATURES_DIR = OUTPUT_DIR / "features"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"

# ---------------------------------------------------------------------------
# Classes  (sub-folder names must match exactly under DATASET_ROOT)
# ---------------------------------------------------------------------------
CLASSES = [
    "cardiovascular_disease",
    "cardiomyopathy",
    "congenital_heart_disease",
    "normal",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Image / data
# ---------------------------------------------------------------------------
IMAGE_SIZE   = (224, 224)   # HxW expected by ResNet-50
TRAIN_SPLIT  = 0.80
RANDOM_SEED  = 42

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE     = 16
NUM_EPOCHS     = 40
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
LR_STEP_SIZE   = 10          # StepLR period
LR_GAMMA       = 0.5         # StepLR decay factor
EARLY_STOP_PATIENCE = 8      # epochs without validation improvement

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
NLM_H              = 0.08    # non-local means filter strength (float image [0,1])
NLM_PATCH_SIZE     = 5
NLM_PATCH_DISTANCE = 6
CLAHE_CLIP_LIMIT   = 0.03
CLAHE_KERNEL_SIZE  = (8, 8)
MULTIOTSU_CLASSES  = 3       # number of Otsu thresholds

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
RESNET_FEATURE_DIM  = 2048   # output of ResNet-50 global-average-pool

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
