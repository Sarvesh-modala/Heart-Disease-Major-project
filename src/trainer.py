"""
trainer.py
----------
Training loop with:
  • Differential learning rates (backbone vs head)
  • Phased training: frozen backbone → unfreeze layer4 → full fine-tune
  • StepLR scheduler
  • Early stopping on validation loss
  • Checkpoint saving for the best validation accuracy
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE, EARLY_STOP_PATIENCE, LEARNING_RATE,
    LR_GAMMA, LR_STEP_SIZE, NUM_EPOCHS, WEIGHT_DECAY,
)
from src.model import HeartMRIClassifier, save_model

logger = logging.getLogger(__name__)


# ===========================================================================
# Helper: one epoch pass
# ===========================================================================

def _run_epoch(
    model: HeartMRIClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    phase: str,
) -> Tuple[float, float]:
    """
    Run one training or evaluation epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float  (0–1)
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss   = 0.0
    correct        = 0
    total          = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                # Gradient clipping prevents exploding gradients during fine-tune
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ===========================================================================
# Main training function
# ===========================================================================

def train_model(
    model: HeartMRIClassifier,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    num_epochs:   int = NUM_EPOCHS,
    lr:           float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    checkpoint_name: str = "best_model.pth",
) -> Dict[str, List[float]]:
    """
    Train the HeartMRIClassifier with a 3-phase schedule:

    Phase 1  (epochs 1 – freeze_epochs):
        Backbone frozen, train classification head only.
        High LR is safe since only the head changes.

    Phase 2  (epochs freeze_epochs – unfreeze_all_epoch):
        Unfreeze layer4 with a 10× smaller LR to gently adapt features.

    Phase 3  (epochs unfreeze_all_epoch – num_epochs):
        Unfreeze full backbone with 100× smaller LR for deep fine-tuning.

    Parameters
    ----------
    model           : HeartMRIClassifier instance (on DEVICE)
    train_loader    : DataLoader for training split
    test_loader     : DataLoader for test/validation split
    num_epochs      : total training epochs
    lr              : base learning rate (for head)
    weight_decay    : L2 regularisation
    checkpoint_name : filename for the best-model checkpoint

    Returns
    -------
    history : dict with keys train_loss, train_acc, val_loss, val_acc
    """

    # Phase boundaries
    freeze_epochs      = max(5,  num_epochs // 8)
    unfreeze_all_epoch = max(10, num_epochs // 4)

    # ------------------------------------------------------------------ #
    # Criterion — label-smoothed cross-entropy is more robust than plain CE
    # ------------------------------------------------------------------ #
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)

    # ------------------------------------------------------------------ #
    # Optimiser — separate param groups for backbone vs head
    # ------------------------------------------------------------------ #
    model.freeze_backbone()
    optimizer = AdamW(
        [{"params": model.head.parameters(), "lr": lr}],
        weight_decay=weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    # ------------------------------------------------------------------ #
    # History + early stopping state
    # ------------------------------------------------------------------ #
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc   = 0.0
    epochs_no_improve = 0

    model.to(DEVICE)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ---- Phase transitions ---------------------------------------- #
        if epoch == freeze_epochs + 1:
            logger.info("Epoch %d: unfreezing layer4 for fine-tuning.", epoch)
            model.unfreeze_backbone(layer4_only=True)
            # Add backbone params with a much smaller lr
            optimizer.add_param_group({
                "params": model.backbone.parameters(),
                "lr": lr / 10.0,
            })

        if epoch == unfreeze_all_epoch + 1:
            logger.info("Epoch %d: unfreezing full backbone.", epoch)
            model.unfreeze_backbone(layer4_only=False)
            # Lower the backbone group lr even further
            for pg in optimizer.param_groups:
                if pg.get("is_backbone"):
                    pg["lr"] = lr / 100.0
            # Mark backbone group so we can find it above
            optimizer.param_groups[-1]["is_backbone"] = True

        # ---- Train ---------------------------------------------------- #
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, phase="train"
        )
        # ---- Validate ------------------------------------------------- #
        val_loss, val_acc = _run_epoch(
            model, test_loader, criterion, optimizer=None, phase="val"
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch [%3d/%3d]  train loss=%.4f acc=%.4f  "
            "val loss=%.4f acc=%.4f  (%.1fs)",
            epoch, num_epochs,
            train_loss, train_acc,
            val_loss,   val_acc,
            elapsed,
        )

        # ---- Checkpoint best model ------------------------------------ #
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, checkpoint_name)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ---- Early stopping ------------------------------------------ #
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                EARLY_STOP_PATIENCE,
            )
            break

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
    return history
