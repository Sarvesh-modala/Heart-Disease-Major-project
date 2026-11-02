"""
model.py
--------
ResNet-50 Transfer Learning Classifier for Heart MRI.

Architecture
------------
  ImageNet-pretrained ResNet-50 backbone
      ↓  (freeze early layers, fine-tune later layers)
  Custom classification head:
      GlobalAvgPool(2048) → Dropout(0.4) → FC(512) → ReLU → Dropout(0.3)
      → FC(NUM_CLASSES)

The backbone is frozen for the first `freeze_epochs` training epochs, then
unfrozen with a lower learning rate (differential learning rates) so the
pretrained features are not destroyed by the new task gradient.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from src.config import DEVICE, MODEL_DIR, NUM_CLASSES

logger = logging.getLogger(__name__)


class HeartMRIClassifier(nn.Module):
    """
    ResNet-50 backbone + custom classification head.

    Parameters
    ----------
    num_classes   : number of output classes (default: NUM_CLASSES = 4)
    dropout_rate  : dropout probability in the classification head
    pretrained    : load ImageNet weights for the backbone
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # Backbone
        # ------------------------------------------------------------------ #
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Keep everything except the original FC classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # ↑ output shape: (B, 2048, 1, 1) after global-average-pool

        # ------------------------------------------------------------------ #
        # Custom head
        # ------------------------------------------------------------------ #
        self.head = nn.Sequential(
            nn.Flatten(),               # (B, 2048)
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(512, num_classes),
        )

        # Initialise head weights with Kaiming normal (good for ReLU)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.backbone(x)   # (B, 2048, 1, 1)
        logits = self.head(feats)   # (B, num_classes)
        return logits

    # ---------------------------------------------------------------------- #
    # Convenience helpers
    # ---------------------------------------------------------------------- #

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (train head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_backbone(self, layer4_only: bool = True) -> None:
        """
        Unfreeze backbone for fine-tuning.

        layer4_only=True  → only unfreeze the last residual block (layer4)
                            and the final BatchNorm; safer starting point.
        layer4_only=False → unfreeze the whole backbone.
        """
        if layer4_only:
            # children of nn.Sequential wrapping backbone:
            # 0: conv1, 1: bn1, 2: relu, 3: maxpool,
            # 4: layer1, 5: layer2, 6: layer3, 7: layer4, 8: avgpool
            for i, child in enumerate(self.backbone.children()):
                for param in child.parameters():
                    param.requires_grad = (i >= 7)   # layer4 + avgpool
            logger.info("Backbone partially unfrozen (layer4 + avgpool).")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Full backbone unfrozen.")

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


# ===========================================================================
# Serialisation helpers
# ===========================================================================

def save_model(model: HeartMRIClassifier, filename: str = "best_model.pth") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    torch.save(model.state_dict(), path)
    logger.info("Model saved → %s", path)
    return path


def load_model(
    path: Path,
    num_classes: int = NUM_CLASSES,
    device: str = DEVICE,
) -> HeartMRIClassifier:
    model = HeartMRIClassifier(num_classes=num_classes, pretrained=False)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s", path)
    return model
