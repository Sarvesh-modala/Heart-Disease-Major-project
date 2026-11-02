#!/usr/bin/env python3
"""
organize_dataset.py
-------------------
Walk the raw Kaggle dataset folder (Normal/Directory_N/series.../*.jpg) and
create symlinks in data/heart_mri/<class>/ with unique flattened names so
filenames from different patients/series don't collide.

Source layout
-------------
<src>/
  Normal/Directory_N/<series>/img*.jpg
  Sick/Directory_N/<series>/img*.jpg

Output layout
-------------
data/heart_mri/
  normal/<Directory_N>__<series>__<filename>
  cardiomyopathy/<Directory_N>__<series>__<filename>
  cardiovascular_disease/<Directory_N>__<series>__<filename>
  congenital_heart_disease/   (left empty — reserved class slot)

Sick directories are split patient-wise so all images from one patient stay
under a single label (medically correct — no patient leakage between labels).
"""

import argparse
import sys
from pathlib import Path

# All 16 Normal directories → normal class.
NORMAL_DIRS = {f"Directory_{i}" for i in range(1, 17)}

# 10 Sick directories split patient-wise into two labels.
# Balanced so total image counts are within ~70 of each other.
CARDIOMYOPATHY_DIRS         = {"Directory_17", "Directory_19", "Directory_20",
                               "Directory_23", "Directory_25"}
CARDIOVASCULAR_DISEASE_DIRS = {"Directory_18", "Directory_21", "Directory_22",
                               "Directory_24", "Directory_26"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def classify_directory(parent_class: str, dir_name: str) -> str | None:
    """Return target class name, or None to skip."""
    if parent_class == "Normal" and dir_name in NORMAL_DIRS:
        return "normal"
    if parent_class == "Sick":
        if dir_name in CARDIOMYOPATHY_DIRS:
            return "cardiomyopathy"
        if dir_name in CARDIOVASCULAR_DISEASE_DIRS:
            return "cardiovascular_disease"
    return None


def organize(src_root: Path, dst_root: Path, copy: bool = False) -> None:
    if not src_root.exists():
        sys.exit(f"Source dataset not found: {src_root}")

    # Ensure all 4 destination class dirs exist (congenital stays empty).
    for cls in ("cardiovascular_disease", "cardiomyopathy",
                "congenital_heart_disease", "normal"):
        (dst_root / cls).mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}

    for parent_class in ("Normal", "Sick"):
        parent_dir = src_root / parent_class
        if not parent_dir.is_dir():
            print(f"Warning: {parent_dir} missing — skipping.", file=sys.stderr)
            continue

        for patient_dir in sorted(parent_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            target_cls = classify_directory(parent_class, patient_dir.name)
            if target_cls is None:
                print(f"  Skipping unmapped dir: {parent_class}/{patient_dir.name}")
                continue

            dst_class_dir = dst_root / target_cls

            for img in patient_dir.rglob("*"):
                if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                    continue

                rel = img.relative_to(patient_dir)
                flat_name = f"{patient_dir.name}__" + "__".join(rel.parts)
                dst = dst_class_dir / flat_name

                if dst.exists() or dst.is_symlink():
                    continue

                if copy:
                    import shutil
                    shutil.copy2(img, dst)
                else:
                    dst.symlink_to(img.resolve())
                totals[target_cls] = totals.get(target_cls, 0) + 1

    print("\n=== Done ===")
    for cls in ("cardiovascular_disease", "cardiomyopathy",
                "congenital_heart_disease", "normal"):
        n = totals.get(cls, 0)
        print(f"  {cls:28s} → {n} files")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="Path to 'Data Set MRI images' folder containing Normal/ and Sick/")
    ap.add_argument("--dst", default=Path("data/heart_mri"), type=Path,
                    help="Destination root (default: data/heart_mri)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of symlinking (uses disk space)")
    args = ap.parse_args()

    organize(args.src.expanduser(), args.dst.expanduser(), copy=args.copy)


if __name__ == "__main__":
    main()
