"""
evaluator.py
------------
Model evaluation utilities producing:
  • Classification report (precision, recall, F1, support)
  • Confusion matrix (normalised heatmap)
  • Per-class ROC curves with AUC
  • Training history curves (loss / accuracy)

All plots are saved as high-resolution PNGs under PLOT_DIR.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CLASSES, DEVICE, NUM_CLASSES, PLOT_DIR

logger = logging.getLogger(__name__)
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


# ===========================================================================
# Prediction collection
# ===========================================================================

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run *loader* through *model* and collect ground-truth labels,
    predicted class indices, and softmax probability vectors.

    Returns
    -------
    y_true  : (N,)       int array
    y_pred  : (N,)       int array
    y_proba : (N, C)     float32 probability array
    """
    model.eval().to(DEVICE)
    all_labels, all_preds, all_proba = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(DEVICE)
        logits = model(images)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(proba, axis=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds)
        all_proba.append(proba)

    y_true  = np.concatenate(all_labels, axis=0)
    y_pred  = np.concatenate(all_preds,  axis=0)
    y_proba = np.concatenate(all_proba,  axis=0)
    return y_true, y_pred, y_proba


# ===========================================================================
# Classification report
# ===========================================================================

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_txt: bool = True,
) -> str:
    """Print and optionally save the sklearn classification report."""
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    if save_txt:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        out = PLOT_DIR / "classification_report.txt"
        out.write_text(report)
        logger.info("Classification report saved → %s", out)

    return report


# ===========================================================================
# Confusion matrix
# ===========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalise: bool = True,
    filename: str = "confusion_matrix.png",
) -> None:
    """
    Plot a (optionally normalised) confusion matrix heatmap.
    Normalisation divides each row by the true-class total, showing recall
    per class rather than raw counts — useful when class sizes differ.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    norm   = "true" if normalise else None
    cm     = confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(CLASSES))),
        normalize=norm,
    )
    fmt    = ".2f" if normalise else "d"
    title  = "Normalised Confusion Matrix" if normalise else "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label",      fontsize=13)
    ax.set_title(title, fontsize=15, pad=12)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out = PLOT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out)


# ===========================================================================
# ROC curves
# ===========================================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    filename: str = "roc_curves.png",
) -> None:
    """
    Plot one ROC curve per class (One-vs-Rest) plus a macro-average curve.
    AUC scores are annotated in the legend.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Binarise labels for one-vs-rest ROC
    y_bin   = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    colours = plt.cm.tab10(np.linspace(0, 0.6, NUM_CLASSES))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC = 0.50)")

    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros(200)
    n_active = 0

    for i, (cls_name, colour) in enumerate(zip(CLASSES, colours)):
        # Skip classes with no positive samples in y_true (e.g. reserved slots)
        if y_bin[:, i].sum() == 0:
            ax.plot([], [], color=colour, lw=2.0,
                    label=f"{cls_name}  (no test samples)")
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=colour, lw=2.0,
                label=f"{cls_name}  (AUC = {auc:.3f})")
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        n_active += 1

    # Macro-average over active (non-empty) classes only
    if n_active > 0:
        mean_tpr /= n_active
        active_idx = [i for i in range(NUM_CLASSES) if y_bin[:, i].sum() > 0]
        macro_auc = roc_auc_score(
            y_bin[:, active_idx], y_proba[:, active_idx],
            average="macro", multi_class="ovr",
        )
    else:
        macro_auc = 0.0
    ax.plot(all_fpr, mean_tpr, "k-", lw=2.5,
            label=f"Macro-average  (AUC = {macro_auc:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title("ROC Curves — One-vs-Rest (per class)", fontsize=15)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = PLOT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved → %s", out)


# ===========================================================================
# Training history curves
# ===========================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    filename: str = "training_history.png",
) -> None:
    """
    Plot loss and accuracy curves for training and validation split.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train loss",  lw=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val loss",    lw=2, linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train acc",  lw=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val acc",    lw=2, linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = PLOT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training history saved → %s", out)


# ===========================================================================
# Master evaluation function
# ===========================================================================

def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    history: Optional[Dict[str, List[float]]] = None,
) -> None:
    """
    Run all evaluation routines in one call.

    Produces:
      outputs/plots/classification_report.txt
      outputs/plots/confusion_matrix.png
      outputs/plots/roc_curves.png
      outputs/plots/training_history.png  (only if history is supplied)
    """
    y_true, y_pred, y_proba = collect_predictions(model, loader)

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true, y_proba)

    if history:
        plot_training_history(history)

    logger.info("All evaluation artefacts saved under %s", PLOT_DIR)
