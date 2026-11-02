#!/usr/bin/env python3
"""
make_split.py
-------------
Materialise the 80/20 stratified train/test split on disk so it's
visible and inspectable. Uses the same RANDOM_SEED as src/data_loader.py
so the split matches what training will see.

Output
------
data/heart_mri_split/
    train/
        cardiovascular_disease/   (symlinks)
        cardiomyopathy/
        congenital_heart_disease/ (empty if source class is empty)
        normal/
    test/
        ...same...
    manifest_train.txt   — one "<class>/<filename>" per line
    manifest_test.txt
"""

import argparse
import random
import sys
from pathlib import Path

CLASSES = [
    "cardiovascular_disease",
    "cardiomyopathy",
    "congenital_heart_disease",
    "normal",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=Path("data/heart_mri"), type=Path,
                    help="Source class-folder root")
    ap.add_argument("--dst", default=Path("data/heart_mri_split"), type=Path,
                    help="Output split root")
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"Source not found: {args.src}")

    rng = random.Random(args.seed)
    train_lines: list[str] = []
    test_lines:  list[str] = []
    counts: dict[str, tuple[int, int]] = {}

    for cls in CLASSES:
        src_cls_dir = args.src / cls
        train_dir = args.dst / "train" / cls
        test_dir  = args.dst / "test"  / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        if not src_cls_dir.is_dir():
            counts[cls] = (0, 0)
            continue

        files = sorted(p for p in src_cls_dir.iterdir()
                       if p.suffix.lower() in IMG_EXTS or p.is_symlink())
        if not files:
            counts[cls] = (0, 0)
            continue

        shuffled = files[:]
        rng.shuffle(shuffled)
        split_at = max(1, int(len(shuffled) * args.train_ratio))
        train_imgs = shuffled[:split_at]
        test_imgs  = shuffled[split_at:]

        for p in train_imgs:
            link = train_dir / p.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(p.resolve())
            train_lines.append(f"{cls}/{p.name}")

        for p in test_imgs:
            link = test_dir / p.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(p.resolve())
            test_lines.append(f"{cls}/{p.name}")

        counts[cls] = (len(train_imgs), len(test_imgs))

    (args.dst / "manifest_train.txt").write_text("\n".join(train_lines) + "\n")
    (args.dst / "manifest_test.txt").write_text("\n".join(test_lines)  + "\n")

    print("\n=== Split summary (seed=%d, train=%.0f%%) ===" %
          (args.seed, args.train_ratio * 100))
    print(f"  {'class':28s} {'train':>8s} {'test':>8s}  {'total':>8s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8}  {'-'*8}")
    grand_train = grand_test = 0
    for cls in CLASSES:
        tr, te = counts.get(cls, (0, 0))
        grand_train += tr
        grand_test  += te
        print(f"  {cls:28s} {tr:>8d} {te:>8d}  {tr+te:>8d}")
    print(f"  {'-'*28} {'-'*8} {'-'*8}  {'-'*8}")
    print(f"  {'TOTAL':28s} {grand_train:>8d} {grand_test:>8d}  "
          f"{grand_train+grand_test:>8d}")
    print(f"\nManifests:\n  {args.dst}/manifest_train.txt"
          f"\n  {args.dst}/manifest_test.txt")
    print(f"\nSymlinked layout under: {args.dst}/")


if __name__ == "__main__":
    main()
