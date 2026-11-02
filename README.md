# Heart MRI Classification

Classifies cardiac MRI images into heart conditions using a hybrid pipeline of classical computer vision and deep learning, with a Streamlit web demo for interactive predictions.

| Class | Description | Images |
|---|---|---|
| `cardiovascular_disease` | Coronary artery disease, hypertension-related changes | 10,962 |
| `cardiomyopathy` | Hypertrophic / dilated myocardial abnormalities | 10,895 |
| `normal` | Healthy heart morphology | 37,566 |
| `congenital_heart_disease` | Structural defects present from birth | 0 (reserved) |

> The model has 4 output slots, but the Kaggle dataset contains no congenital heart disease scans, so that class is currently empty. Results below are reported over the 3 classes that have data.

---

## Results

Evaluated on the held-out test split (11,886 images, 20 %):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| `cardiovascular_disease` | 0.9643 | 0.9850 | 0.9745 | 2,193 |
| `cardiomyopathy` | 0.9807 | 0.9807 | 0.9807 | 2,179 |
| `normal` | 0.9930 | 0.9868 | 0.9899 | 7,514 |
| **Overall accuracy** | | | **98.54 %** | 11,886 |

![Confusion matrix](outputs/plots/confusion_matrix_3class.png)
![ROC curves](outputs/plots/roc_curves.png)
![Preprocessing stages](outputs/plots/preprocessing_stages.png)

---

## Architecture Overview

```
Raw MRI
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Preprocessing Pipeline                             │
│  1. Non-rigid B-spline registration (SimpleITK)     │
│  2. Non-local means denoising (skimage)             │
│  3. Intensity standardisation (percentile + z-score)│
│  4a. CLAHE contrast enhancement                     │
│  4b. DnCNN deep learning enhancer                   │
│  5. Multilevel Otsu thresholding (soft mask blend)  │
└─────────────────────────────────────────────────────┘
  │
  ├──► HOG Feature Vectors  →  saved as .npy
  │
  └──► ResNet-50 Classifier (transfer learning)
           ├─ Frozen backbone  (phase 1)
           ├─ Fine-tune layer4 (phase 2)
           └─ Full fine-tune   (phase 3)
                │
                ▼
           Evaluation: confusion matrix · ROC · F1
```

---

## 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate.bat       # Windows

# Install packages
pip install -r requirements.txt
```

> **GPU (recommended):** Install the CUDA-enabled PyTorch build first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
> then run `pip install -r requirements.txt`.

---

## 2. Download the Dataset

The MRI images are **not included in this repository** (about 2.7 GB). Download them from Kaggle:  
**https://www.kaggle.com/datasets/danialsharifrazi/hypertrophic-cardiomyopathy-dataset**

**Option A: automatic** (needs a Kaggle API token in `~/.kaggle/kaggle.json`)

```bash
bash setup_dataset.sh
```

**Option B: manual**

1. Download and unzip the archive. It contains `Normal/Directory_N/...` and `Sick/Directory_N/...` folders.
2. Organise the images into class folders. Sick patients are split patient-wise between `cardiomyopathy` and `cardiovascular_disease`, so no patient appears in both classes:

```bash
python scripts/organize_dataset.py --src "/path/to/Data Set MRI images"   # add --copy to copy instead of symlink
python scripts/make_split.py                                              # 80/20 stratified split, seed 42
```

This produces:

```
data/
├── heart_mri/                 ← all images, one folder per class
│   ├── cardiovascular_disease/
│   ├── cardiomyopathy/
│   ├── congenital_heart_disease/   (empty)
│   └── normal/
└── heart_mri_split/           ← train/ and test/ with the same class folders
    ├── manifest_train.txt     ← included in the repo, so the exact split is reproducible
    └── manifest_test.txt
```

---

## 3. Run Training

```bash
# Full pipeline (recommended defaults)
python main.py train

# Custom options
python main.py train \
  --epochs 60 \
  --batch-size 32 \
  --lr 5e-5 \
  --skip-registration        # faster — skip non-rigid registration
```

**What happens during training:**

1. Images are discovered and stratified-split 80 % / 20 % (train / test).
2. A `WeightedRandomSampler` balances minority classes in every batch.
3. HOG feature vectors are extracted and saved.
4. ResNet-50 deep features are extracted (ImageNet weights).
5. The classifier trains in 3 phases:
   - Phase 1 — backbone frozen, head only
   - Phase 2 — layer4 + head fine-tuned
   - Phase 3 — full network fine-tuned
6. Best validation-accuracy checkpoint is saved automatically.
7. Final evaluation artefacts are generated.

Training log is written to `run.log`.

---

## 4. Run Inference

```bash
# Single image
python main.py infer --image data/heart_mri/normal/scan_001.png

# Entire directory
python main.py infer --dir data/unseen_scans/

# Use a specific checkpoint
python main.py infer --image scan.png --checkpoint outputs/models/best_model.pth

# Skip preprocessing (use raw image directly)
python main.py infer --image scan.png --no-preprocess
```

**From Python:**

```python
from src.inference import predict_single, print_prediction

result = predict_single("data/heart_mri/normal/scan_001.png")
print_prediction(result)
# Output:
# ==================================================
#   Predicted class  : normal
#   Confidence       : 97.42%
#   All probabilities:
#     cardiovascular_disease         0.0081  |##
#     cardiomyopathy                 0.0135  |####
#     congenital_heart_disease       0.0042  |#
#     normal                         0.9742  |##############################
# ==================================================
```

---

## 5. Web Demo (Streamlit)

```bash
pip install streamlit
streamlit run app.py
```

Open http://localhost:8501, upload an MRI image (or pick a sample), and the page shows the predicted class, confidence, per-class probabilities, and the preprocessing stages.

`run.sh` / `run.bat` start the Streamlit backend together with the separate `heart-mri-web` frontend, if you have it checked out next to this folder.

---

## 6. Standalone Feature Extraction

```bash
# Extract HOG + deep features for both train and test splits
python main.py features --split both

# Train split only
python main.py features --split train
```

Feature files are saved under `outputs/features/`:
- `train_hog_X.npy` / `train_hog_y.npy`
- `test_hog_X.npy`  / `test_hog_y.npy`
- `train_deep_X.npy` / `train_deep_y.npy`
- `test_deep_X.npy`  / `test_deep_y.npy`

---

## 7. Re-evaluate a Saved Checkpoint

```bash
python main.py evaluate --checkpoint outputs/models/best_model.pth
```

---

## 8. Output Folder Structure

```
outputs/
├── models/
│   └── best_model.pth              ← Best validation checkpoint
├── features/
│   ├── train_hog_X.npy             ← HOG feature matrix (train)
│   ├── train_hog_y.npy             ← Labels (train)
│   ├── test_hog_X.npy
│   ├── test_hog_y.npy
│   ├── train_deep_X.npy            ← ResNet-50 deep features (train)
│   ├── train_deep_y.npy
│   ├── test_deep_X.npy
│   └── test_deep_y.npy
└── plots/
    ├── classification_report.txt   ← Precision / Recall / F1
    ├── confusion_matrix.png        ← Normalised heatmap
    ├── confusion_matrix_3class.png ← Confusion matrix over the 3 classes with data
    ├── roc_curves.png              ← Per-class ROC + macro AUC
    └── preprocessing_stages.png    ← Example of each preprocessing step
run.log                             ← Full training log
```

> **Not in the repository:** `outputs/models/`, `outputs/features/` and `outputs/preprocessed/` are too large for GitHub (up to 1.1 GB per file). Run `python main.py train` to regenerate them. Only `outputs/plots/` and a few sample images in `outputs/preprocessed_demo/` are committed.

---

## 9. Configuration

All tunable constants are in `src/config.py`:

| Constant | Default | Meaning |
|---|---|---|
| `NUM_EPOCHS` | 40 | Training epochs |
| `BATCH_SIZE` | 16 | Images per batch |
| `LEARNING_RATE` | 1e-4 | Head LR (backbone uses 10× / 100× smaller) |
| `TRAIN_SPLIT` | 0.80 | Fraction of data used for training |
| `EARLY_STOP_PATIENCE` | 8 | Epochs without improvement before stopping |
| `CLAHE_CLIP_LIMIT` | 0.03 | CLAHE contrast limit |
| `MULTIOTSU_CLASSES` | 3 | Number of Otsu segmentation regions |
| `HOG_ORIENTATIONS` | 9 | HOG gradient orientation bins |
| `NLM_H` | 0.08 | Non-local means filter strength |

---

## 10. Preprocessing Pipeline Details

| Step | Method | Purpose |
|---|---|---|
| Non-rigid registration | SimpleITK B-spline FFD | Align hearts to a common atlas frame |
| NLM denoising | `skimage.restoration.denoise_nl_means` | Remove scanner noise while preserving myocardial boundaries |
| Intensity standardisation | Percentile clip + min-max rescale | Remove inter-scanner intensity bias |
| CLAHE | `skimage.exposure.equalize_adapthist` | Enhance local contrast in low-signal regions |
| DL enhancement | DnCNN residual CNN (`src/preprocessing.py`) | Learned noise subtraction / sharpening |
| Multilevel Otsu | `skimage.filters.threshold_multiotsu` | Soft tissue / background separation |

---

## 11. Requirements

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scipy>=1.11.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
SimpleITK>=2.3.0
opencv-python>=4.8.0
tqdm>=4.65.0
pandas>=2.0.0
joblib>=1.3.0
PyWavelets>=1.4.0
```

For the web demo, also install `streamlit`.

---

## Citation / Dataset

> Sharifrazi, D. (2023). *Hypertrophic Cardiomyopathy Dataset*.  
> Kaggle. https://www.kaggle.com/datasets/danialsharifrazi/hypertrophic-cardiomyopathy-dataset
