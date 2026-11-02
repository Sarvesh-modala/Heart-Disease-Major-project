#!/usr/bin/env python3
"""
visualize_preprocessing.py
--------------------------
Pick one sample MRI from each class and produce a panel showing every
intermediate preprocessing stage side-by-side. Saves to
outputs/plots/preprocessing_stages.png — drop this on a slide for the demo.
"""

# Match main.py threading guard.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASSES, PLOT_DIR
from src.preprocessing import (
    apply_clahe, dl_enhance, intensity_standardise,
    load_image_gray, multilevel_threshold, nlm_denoise,
)


STAGES = [
    ("Raw",                 "raw"),
    ("NLM denoise",         "nlm"),
    ("Intensity standardise", "intensity"),
    ("CLAHE",               "clahe"),
    ("DnCNN enhance",       "dncnn"),
    ("Multi-Otsu blend",    "otsu"),
]


def run_pipeline(path: Path) -> dict:
    """Run each stage and return {stage_key: float32 image}."""
    out = {}
    img = load_image_gray(path)
    out["raw"]       = img.copy()
    img = nlm_denoise(img);          out["nlm"]       = img.copy()
    img = intensity_standardise(img); out["intensity"] = img.copy()
    img = apply_clahe(img);          out["clahe"]     = img.copy()
    img = dl_enhance(img);           out["dncnn"]     = img.copy()
    img = multilevel_threshold(img); out["otsu"]      = img.copy()
    return out


def pick_sample(class_dir: Path, seed: int = 0) -> Path:
    rng = random.Random(seed)
    files = sorted(p for p in class_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"} or p.is_symlink())
    if not files:
        raise FileNotFoundError(f"No images in {class_dir}")
    return rng.choice(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/heart_mri", type=Path)
    ap.add_argument("--out", default=PLOT_DIR / "preprocessing_stages.png", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Pick one sample per non-empty class
    samples: list[tuple[str, Path, dict]] = []
    for cls in CLASSES:
        cls_dir = args.data_root / cls
        if not cls_dir.is_dir() or not any(cls_dir.iterdir()):
            print(f"  skipping empty class: {cls}")
            continue
        path = pick_sample(cls_dir, seed=args.seed)
        print(f"  {cls:28s} → {path.name}")
        print("    running pipeline...", flush=True)
        stages = run_pipeline(path)
        samples.append((cls, path, stages))

    if not samples:
        raise SystemExit("No samples found.")

    n_rows = len(samples)
    n_cols = len(STAGES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for r, (cls, path, stages) in enumerate(samples):
        for c, (title, key) in enumerate(STAGES):
            ax = axes[r, c]
            ax.imshow(stages[key], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(cls.replace("_", "\n"), fontsize=10, rotation=0,
                              labelpad=50, va="center", ha="right")

    fig.suptitle("Preprocessing pipeline — stage-by-stage", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
