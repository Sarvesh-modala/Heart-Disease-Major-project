#!/usr/bin/env python3
"""
main.py
-------
Command-line entry-point for the Heart MRI Classification project.

Commands
--------
  train      Full training pipeline: preprocess → extract features → train → evaluate
  infer      Predict a single image or all images in a directory
  features   Extract and save HOG / deep feature vectors (no training)
  evaluate   Re-evaluate a saved checkpoint on the test split

Examples
--------
  python main.py train
  python main.py train --epochs 60 --batch-size 32 --skip-registration
  python main.py infer --image data/test_scan.png
  python main.py infer --dir  data/unseen_scans/
  python main.py features --split train
  python main.py evaluate --checkpoint outputs/models/best_model.pth
"""

import os
# Avoid OMP/torch + skimage lazy-import deadlock seen with torch 2.11 + Py3.13.
# Must be set before any numerical libraries import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (must happen before any src import so all loggers inherit it)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports (after logging is configured)
# ---------------------------------------------------------------------------

def _train(args: argparse.Namespace) -> None:
    """Full training pipeline."""
    import torch

    from src.config import (
        DATASET_ROOT, DEVICE, FEATURES_DIR, NUM_EPOCHS,
        BATCH_SIZE, PREPROCESSED_DIR,
    )
    from src.data_loader import get_dataloaders
    from src.evaluator import full_evaluation
    from src.feature_extraction import (
        build_hog_feature_matrix,
        extract_deep_features,
    )
    from src.model import HeartMRIClassifier
    from src.trainer import train_model

    logger.info("=== Heart MRI Classifier — Training Pipeline ===")
    logger.info("Device: %s", DEVICE)

    # ------------------------------------------------------------------ #
    # 1. Data loading
    # ------------------------------------------------------------------ #
    logger.info("Loading dataset from %s ...", DATASET_ROOT)
    train_loader, test_loader, train_samples, test_samples = get_dataloaders(
        root=Path(args.dataset_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------ #
    # 2. HOG feature extraction (classical)
    # ------------------------------------------------------------------ #
    if not args.skip_hog:
        logger.info("Extracting HOG features ...")
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)

        build_hog_feature_matrix(
            train_samples,
            preprocessed_dir=PREPROCESSED_DIR if not args.skip_registration else None,
            save_path=FEATURES_DIR / "train_hog",
        )
        build_hog_feature_matrix(
            test_samples,
            preprocessed_dir=PREPROCESSED_DIR if not args.skip_registration else None,
            save_path=FEATURES_DIR / "test_hog",
        )
        logger.info("HOG features saved under %s", FEATURES_DIR)

    # ------------------------------------------------------------------ #
    # 3. Build ResNet model
    # ------------------------------------------------------------------ #
    logger.info("Initialising HeartMRIClassifier (ResNet-50 backbone) ...")
    model = HeartMRIClassifier(pretrained=True).to(DEVICE)

    # ------------------------------------------------------------------ #
    # 4. Deep feature extraction (before fine-tuning — ImageNet weights)
    # ------------------------------------------------------------------ #
    if not args.skip_deep_features:
        logger.info("Extracting deep (ResNet-50) features ...")
        extract_deep_features(
            train_loader,
            save_path=FEATURES_DIR / "train_deep",
        )
        extract_deep_features(
            test_loader,
            save_path=FEATURES_DIR / "test_deep",
        )

    # ------------------------------------------------------------------ #
    # 5. Training
    # ------------------------------------------------------------------ #
    logger.info("Starting training for %d epochs ...", args.epochs)
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        checkpoint_name="best_model.pth",
    )

    # ------------------------------------------------------------------ #
    # 6. Full evaluation on test set
    # ------------------------------------------------------------------ #
    logger.info("Running final evaluation ...")
    # Reload best checkpoint for evaluation
    from src.model import load_model, MODEL_DIR
    best_model = load_model(MODEL_DIR / "best_model.pth")
    full_evaluation(best_model, test_loader, history=history)

    logger.info("=== Training complete. Outputs: outputs/ ===")


def _infer(args: argparse.Namespace) -> None:
    """Inference on one image or a whole directory."""
    from src.inference import predict_single, predict_batch, print_prediction

    if args.image:
        result = predict_single(
            args.image,
            model_path=Path(args.checkpoint) if args.checkpoint else None,
            apply_preprocessing=not args.no_preprocess,
        )
        print_prediction(result)

    elif args.dir:
        img_dir = Path(args.dir)
        exts    = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths   = [p for p in img_dir.iterdir() if p.suffix.lower() in exts]
        if not paths:
            logger.error("No image files found in %s", img_dir)
            sys.exit(1)
        results = predict_batch(
            paths,
            model_path=Path(args.checkpoint) if args.checkpoint else None,
            apply_preprocessing=not args.no_preprocess,
        )
        for path, res in zip(paths, results):
            print(f"\n[{path.name}]")
            print_prediction(res)
    else:
        logger.error("Provide --image or --dir for inference.")
        sys.exit(1)


def _features(args: argparse.Namespace) -> None:
    """Standalone feature extraction."""
    from src.config import FEATURES_DIR, PREPROCESSED_DIR
    from src.data_loader import get_dataloaders
    from src.feature_extraction import build_hog_feature_matrix, extract_deep_features

    train_loader, test_loader, train_samples, test_samples = get_dataloaders(
        root=Path(args.dataset_root)
    )

    split = args.split.lower()
    if split in ("train", "both"):
        build_hog_feature_matrix(
            train_samples,
            preprocessed_dir=PREPROCESSED_DIR / "train",
            save_path=FEATURES_DIR / "train_hog",
        )
        extract_deep_features(train_loader, save_path=FEATURES_DIR / "train_deep")
    if split in ("test", "both"):
        build_hog_feature_matrix(
            test_samples,
            preprocessed_dir=PREPROCESSED_DIR / "test",
            save_path=FEATURES_DIR / "test_hog",
        )
        extract_deep_features(test_loader, save_path=FEATURES_DIR / "test_deep")


def _preprocess(args: argparse.Namespace) -> None:
    """Run only the preprocessing pipeline and save outputs to disk."""
    import random
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from src.config import CLASSES, PREPROCESSED_DIR
    from src.preprocessing import preprocess_and_save

    src_root = Path(args.dataset_root)
    if not src_root.exists():
        logger.error("Dataset root not found: %s", src_root)
        sys.exit(1)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    work: list[tuple[Path, Path]] = []
    for cls in CLASSES:
        cls_dir = src_root / cls
        if not cls_dir.is_dir():
            logger.warning("Class folder missing: %s", cls_dir)
            continue
        images = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in exts)
        if args.max_per_class and args.max_per_class > 0:
            rng = random.Random(42)
            rng.shuffle(images)
            images = images[: args.max_per_class]
        out_dir = Path(args.output_dir) / cls
        for img in images:
            work.append((img, out_dir))
        logger.info("Class %-20s queued %d images", cls, len(images))

    if not work:
        logger.error("Nothing to preprocess.")
        sys.exit(1)

    logger.info("Total images to preprocess: %d  →  %s",
                len(work), args.output_dir)

    kwargs = dict(skip_registration=args.skip_registration)

    if args.workers <= 1:
        from tqdm import tqdm
        for img_path, out_dir in tqdm(work, desc="Preprocess"):
            try:
                preprocess_and_save(img_path, out_dir, **kwargs)
            except Exception as exc:
                logger.error("Failed on %s: %s", img_path, exc)
    else:
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(preprocess_and_save, img, out, **kwargs)
                for img, out in work
            ]
            for _ in tqdm(as_completed(futs), total=len(futs), desc="Preprocess"):
                pass

    logger.info("Preprocessing complete.")


def _preprocess(args: argparse.Namespace) -> None:
    """Run only the preprocessing pipeline and save outputs to disk."""
    import random
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from src.config import CLASSES
    from src.preprocessing import preprocess_and_save

    src_root = Path(args.dataset_root)
    if not src_root.exists():
        logger.error("Dataset root not found: %s", src_root)
        sys.exit(1)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    if args.split == "all":
        splits = [None]

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    work: list[tuple[Path, Path]] = []

    def _collect_class_dir(class_dir: Path, out_class_dir: Path) -> None:
        if not class_dir.is_dir():
            logger.warning("Class folder missing: %s", class_dir)
            return
        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in exts or p.is_symlink()
        )
        if args.max_per_class and args.max_per_class > 0:
            rng = random.Random(42)
            rng.shuffle(images)
            images = images[: args.max_per_class]
        for img in images:
            work.append((img, out_class_dir))
        logger.info("  %-30s queued %d images", class_dir.name, len(images))

    if splits == [None]:
        # Flat layout: <root>/<class>/
        logger.info("Collecting from flat layout: %s", src_root)
        for cls in CLASSES:
            _collect_class_dir(src_root / cls,
                               Path(args.output_dir) / cls)
    else:
        # Split layout: <root>/<split>/<class>/
        for split in splits:
            split_root = src_root / split
            logger.info("Collecting split: %s", split_root)
            for cls in CLASSES:
                _collect_class_dir(split_root / cls,
                                   Path(args.output_dir) / split / cls)

    if not work:
        logger.error("Nothing to preprocess.")
        sys.exit(1)

    logger.info("Total images to preprocess: %d  →  %s",
                len(work), args.output_dir)
    if args.skip_registration:
        logger.info("Non-rigid registration: SKIPPED")

    kwargs = dict(skip_registration=args.skip_registration)

    from tqdm import tqdm

    if args.workers <= 1:
        for img_path, out_dir in tqdm(work, desc="Preprocess"):
            try:
                preprocess_and_save(img_path, out_dir, **kwargs)
            except Exception as exc:
                logger.error("Failed on %s: %s", img_path, exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(preprocess_and_save, img, out, **kwargs)
                for img, out in work
            ]
            for _ in tqdm(as_completed(futs), total=len(futs), desc="Preprocess"):
                pass

    logger.info("Preprocessing complete.")


def _evaluate(args: argparse.Namespace) -> None:
    """Re-evaluate a saved checkpoint without retraining."""
    from src.config import DATASET_ROOT
    from src.data_loader import get_dataloaders
    from src.evaluator import full_evaluation
    from src.model import load_model

    _, test_loader, _, _ = get_dataloaders(root=Path(args.dataset_root))
    model = load_model(Path(args.checkpoint))
    full_evaluation(model, test_loader)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Heart MRI Classification — ResNet-50 + Classical Features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----------------------------------------------------------- #
    p_train = sub.add_parser("train", help="Train the classifier")
    p_train.add_argument("--dataset-root",       default="data/heart_mri",
                         help="Root folder of the dataset")
    p_train.add_argument("--epochs",       type=int,   default=NUM_EPOCHS_DEFAULT,
                         help="Total training epochs")
    p_train.add_argument("--batch-size",   type=int,   default=16)
    p_train.add_argument("--lr",           type=float, default=1e-4,
                         help="Base learning rate for the classification head")
    p_train.add_argument("--num-workers",  type=int,   default=4)
    p_train.add_argument("--skip-registration", action="store_true",
                         help="Skip non-rigid registration (faster preprocessing)")
    p_train.add_argument("--skip-hog",          action="store_true",
                         help="Skip HOG feature extraction")
    p_train.add_argument("--skip-deep-features", action="store_true",
                         help="Skip pre-training deep feature extraction")

    # ---- infer ----------------------------------------------------------- #
    p_infer = sub.add_parser("infer", help="Run inference")
    p_infer.add_argument("--image",      help="Path to a single image")
    p_infer.add_argument("--dir",        help="Directory of images")
    p_infer.add_argument("--checkpoint", help="Path to model .pth file")
    p_infer.add_argument("--no-preprocess", action="store_true",
                         help="Skip preprocessing (use raw image)")

    # ---- features -------------------------------------------------------- #
    p_feat = sub.add_parser("features", help="Extract and save feature vectors")
    p_feat.add_argument("--dataset-root", default="data/heart_mri")
    p_feat.add_argument("--split", choices=["train", "test", "both"], default="both")

    # ---- preprocess ------------------------------------------------------ #
    p_pre = sub.add_parser("preprocess",
                           help="Run preprocessing pipeline and save outputs")
    p_pre.add_argument("--dataset-root", default="data/heart_mri_split",
                       help="Either flat 'data/heart_mri' (use --split all) "
                            "or split 'data/heart_mri_split' (use --split train/test/both)")
    p_pre.add_argument("--output-dir",   default="outputs/preprocessed")
    p_pre.add_argument("--split", choices=["train", "test", "both", "all"],
                       default="both")
    p_pre.add_argument("--skip-registration", action="store_true",
                       help="Skip non-rigid registration (much faster)")
    p_pre.add_argument("--max-per-class", type=int, default=0,
                       help="Limit images per class (0 = all). Useful for sampling.")
    p_pre.add_argument("--workers", type=int, default=1,
                       help="Parallel worker processes (1 = serial)")

    # ---- evaluate -------------------------------------------------------- #
    p_eval = sub.add_parser("evaluate", help="Evaluate a saved checkpoint")
    p_eval.add_argument("--checkpoint",   default="outputs/models/best_model.pth")
    p_eval.add_argument("--dataset-root", default="data/heart_mri")

    return parser


# ---------------------------------------------------------------------------
# Import constant after logging init
# ---------------------------------------------------------------------------
try:
    from src.config import NUM_EPOCHS
    NUM_EPOCHS_DEFAULT = NUM_EPOCHS
except Exception:
    NUM_EPOCHS_DEFAULT = 40


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "train":      _train,
        "infer":      _infer,
        "features":   _features,
        "preprocess": _preprocess,
        "evaluate":   _evaluate,
    }
    try:
        dispatch[args.command](args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
