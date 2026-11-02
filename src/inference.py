"""
inference.py
------------
Single-image and batch inference for the trained HeartMRIClassifier.

Usage (from code)
-----------------
    from src.inference import predict_single, predict_batch
    result = predict_single("path/to/mri.png")
    print(result)  # {"class": "cardiomyopathy", "confidence": 0.93, ...}

Usage (CLI)
-----------
    python main.py infer --image path/to/mri.png
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import CLASSES, DEVICE, IMAGE_SIZE, MODEL_DIR
from src.model import HeartMRIClassifier, load_model
from src.preprocessing import preprocess_image

logger = logging.getLogger(__name__)

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ===========================================================================
# Model loading helper
# ===========================================================================

def _load_inference_model(
    model_path: Optional[Path] = None,
) -> HeartMRIClassifier:
    """
    Load the best saved checkpoint.  Falls back to the default path
    outputs/models/best_model.pth if *model_path* is None.
    """
    if model_path is None:
        model_path = MODEL_DIR / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model checkpoint found at {model_path}.\n"
            "Train the model first:  python main.py train"
        )
    return load_model(model_path, device=DEVICE)


# ===========================================================================
# Core prediction function
# ===========================================================================

@torch.no_grad()
def _predict_tensor(
    model: HeartMRIClassifier,
    tensor: torch.Tensor,
) -> Dict[str, object]:
    """
    Run *tensor* (1 × 3 × H × W) through *model* and return a structured
    result dict.
    """
    model.eval()
    logits = model(tensor.to(DEVICE))
    proba  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_idx = int(np.argmax(proba))
    return {
        "class":       CLASSES[predicted_idx],
        "class_index": predicted_idx,
        "confidence":  float(proba[predicted_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, proba)},
    }


# ===========================================================================
# Public API
# ===========================================================================

def predict_single(
    image_path: Union[str, Path],
    model: Optional[HeartMRIClassifier] = None,
    model_path: Optional[Path] = None,
    apply_preprocessing: bool = True,
    skip_registration: bool = True,
) -> Dict[str, object]:
    """
    Predict the cardiac condition class for a single MRI image.

    Parameters
    ----------
    image_path          : path to the MRI image file
    model               : pre-loaded model (if None, loads from model_path)
    model_path          : path to .pth checkpoint
    apply_preprocessing : run the full preprocessing pipeline before inference
    skip_registration   : skip non-rigid registration (faster, usually fine
                          for a single novel image)

    Returns
    -------
    dict with keys:
        class        – predicted class name
        class_index  – integer class index
        confidence   – softmax probability of the predicted class
        probabilities – dict mapping every class name to its probability
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        model = _load_inference_model(model_path)

    if apply_preprocessing:
        arr = preprocess_image(image_path, skip_registration=skip_registration)
        # Convert single-channel float32 to 3-channel PIL for the transform
        arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8, mode="L").convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    tensor = _INFER_TRANSFORM(img).unsqueeze(0)   # (1, 3, H, W)
    result = _predict_tensor(model, tensor)

    # Pretty-print
    logger.info(
        "Prediction for %s → %s (confidence: %.2f%%)",
        image_path.name, result["class"], result["confidence"] * 100,
    )
    return result


def predict_batch(
    image_paths: List[Union[str, Path]],
    model: Optional[HeartMRIClassifier] = None,
    model_path: Optional[Path] = None,
    apply_preprocessing: bool = True,
    skip_registration: bool = True,
) -> List[Dict[str, object]]:
    """
    Predict cardiac conditions for a list of MRI image paths.
    Loads the model once and reuses it for all images.

    Returns
    -------
    List of result dicts (same structure as predict_single).
    """
    if model is None:
        model = _load_inference_model(model_path)

    results = []
    for path in image_paths:
        try:
            res = predict_single(
                path,
                model=model,
                apply_preprocessing=apply_preprocessing,
                skip_registration=skip_registration,
            )
            results.append(res)
        except Exception as exc:
            logger.error("Failed to predict %s: %s", path, exc)
            results.append({"error": str(exc), "path": str(path)})

    return results


def print_prediction(result: Dict[str, object]) -> None:
    """Pretty-print a single prediction result to stdout."""
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return

    print("\n" + "=" * 50)
    print(f"  Predicted class  : {result['class']}")
    print(f"  Confidence       : {result['confidence']:.2%}")
    print("  All probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"    {cls:<30} {prob:.4f}  |{bar}")
    print("=" * 50)
