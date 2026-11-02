"""
scripts/make_confusion_matrix_3class.py
---------------------------------------
Regenerate the confusion-matrix artefact over only the classes that actually
have data, so the Streamlit page shows a 3x3 grid instead of a 4x4 with an
empty congenital_heart_disease row/column.

Reads nothing but the existing checkpoint and the existing test split; writes
outputs/plots/confusion_matrix_3class.png (plus the raw predictions as .npy so
the matrix can be rebuilt without another inference pass). The original
confusion_matrix.png is left untouched.

Run:
    PYTHONPATH=. venv/bin/python scripts/make_confusion_matrix_3class.py
"""

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CLASSES, DEVICE, MODEL_DIR, PLOT_DIR
from src.data_loader import HeartMRIDataset, build_transforms
from src.model import load_model

# Same hidden-class list the Streamlit page uses.
HIDDEN_CLASSES  = {"congenital_heart_disease"}
DISPLAY_IDX     = [i for i, c in enumerate(CLASSES) if c not in HIDDEN_CLASSES]
DISPLAY_CLASSES = [CLASSES[i] for i in DISPLAY_IDX]

TEST_ROOT = Path("data/heart_mri_split/test")
BATCH     = 64

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def collect_test_samples():
    samples = []
    for idx in DISPLAY_IDX:
        cls_dir = TEST_ROOT / CLASSES[idx]
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.name.startswith("."):
                continue
            if p.is_file() or p.is_symlink():
                samples.append((p, idx))
    return samples


def cached_predictions():
    """Reuse the saved predictions so re-rendering doesn't redo inference."""
    t, pr = PLOT_DIR / "cm3_y_true.npy", PLOT_DIR / "cm3_y_pred.npy"
    if "--refresh" in sys.argv or not (t.exists() and pr.exists()):
        return None
    print("Reusing cached predictions (pass --refresh to re-run inference)")
    return np.load(t), np.load(pr)


def run_inference():
    samples = collect_test_samples()
    print(f"Test images: {len(samples)}")
    for idx in DISPLAY_IDX:
        print(f"  {CLASSES[idx]:26s} {sum(1 for _, y in samples if y == idx)}")

    loader = DataLoader(
        HeartMRIDataset(samples, transform=build_transforms(train=False)),
        batch_size=BATCH, shuffle=False, num_workers=0,
    )

    model = load_model(MODEL_DIR / "best_model.pth")
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Inference", unit="batch"):
            logits = model(xb.to(DEVICE))
            y_pred.append(logits.argmax(1).cpu().numpy())
            y_true.append(yb.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PLOT_DIR / "cm3_y_true.npy", y_true)
    np.save(PLOT_DIR / "cm3_y_pred.npy", y_pred)
    return y_true, y_pred


def main() -> None:
    y_true, y_pred = cached_predictions() or run_inference()

    # How many predictions landed on a hidden class (dropped from the 3x3)
    dropped = int(sum(1 for p in y_pred if p not in DISPLAY_IDX))
    print(f"Predictions falling on a hidden class (excluded from the 3x3): {dropped}")

    cm = confusion_matrix(y_true, y_pred, labels=DISPLAY_IDX, normalize="true")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=DISPLAY_CLASSES, yticklabels=DISPLAY_CLASSES,
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)
    ax.set_title("Normalised Confusion Matrix", fontsize=15, pad=12)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out = PLOT_DIR / "confusion_matrix_3class.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")
    print("Matrix (row-normalised):")
    for name, row in zip(DISPLAY_CLASSES, cm):
        print(f"  {name:26s} " + "  ".join(f"{v:.4f}" for v in row))


if __name__ == "__main__":
    main()
