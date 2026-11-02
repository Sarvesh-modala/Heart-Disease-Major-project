#!/usr/bin/env bash
# setup_dataset.sh
# Downloads the Kaggle dataset and organises it into the 4-class folder layout.
# Usage: bash setup_dataset.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_DIR/data/heart_mri"
TMP_DIR="$PROJECT_DIR/data/_tmp_download"

echo ""
echo "=== Heart MRI Dataset Setup ==="
echo ""

# ── Check kaggle credentials ─────────────────────────────────────────────────
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "ERROR: ~/.kaggle/kaggle.json not found."
  echo ""
  echo "  1. Go to https://www.kaggle.com → Settings → API → Create New Token"
  echo "  2. Move the downloaded file:  cp ~/Downloads/kaggle.json ~/.kaggle/"
  echo "  3. Set permissions:           chmod 600 ~/.kaggle/kaggle.json"
  echo "  4. Re-run this script."
  exit 1
fi

# ── Create folders ────────────────────────────────────────────────────────────
mkdir -p "$TMP_DIR"
mkdir -p "$DATA_DIR/cardiovascular_disease"
mkdir -p "$DATA_DIR/cardiomyopathy"
mkdir -p "$DATA_DIR/congenital_heart_disease"
mkdir -p "$DATA_DIR/normal"

echo "Downloading dataset from Kaggle..."
kaggle datasets download \
  -d danialsharifrazi/hypertrophic-cardiomyopathy-dataset \
  -p "$TMP_DIR" \
  --unzip

echo ""
echo "Downloaded files:"
find "$TMP_DIR" -type d | head -40
echo ""

# ── Auto-map whatever folder names Kaggle uses → our 4 class names ───────────
# The script looks for folder names containing keywords and copies images across.

declare -A KEYWORD_MAP
KEYWORD_MAP["cardiovascular"]="cardiovascular_disease"
KEYWORD_MAP["coronary"]="cardiovascular_disease"
KEYWORD_MAP["hypertension"]="cardiovascular_disease"
KEYWORD_MAP["cardiomyopathy"]="cardiomyopathy"
KEYWORD_MAP["hypertrophic"]="cardiomyopathy"
KEYWORD_MAP["dilated"]="cardiomyopathy"
KEYWORD_MAP["congenital"]="congenital_heart_disease"
KEYWORD_MAP["structural"]="congenital_heart_disease"
KEYWORD_MAP["normal"]="normal"
KEYWORD_MAP["healthy"]="normal"

MAPPED=0

for SRC_DIR in "$TMP_DIR"/*/; do
  FOLDER=$(basename "$SRC_DIR" | tr '[:upper:]' '[:lower:]')
  DEST_CLASS=""

  for KEYWORD in "${!KEYWORD_MAP[@]}"; do
    if [[ "$FOLDER" == *"$KEYWORD"* ]]; then
      DEST_CLASS="${KEYWORD_MAP[$KEYWORD]}"
      break
    fi
  done

  if [ -n "$DEST_CLASS" ]; then
    COUNT=$(find "$SRC_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" \) | wc -l | tr -d ' ')
    echo "  Mapping '$FOLDER' → $DEST_CLASS  ($COUNT images)"
    find "$SRC_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.bmp" -o -iname "*.tif" \) \
      -exec cp {} "$DATA_DIR/$DEST_CLASS/" \;
    MAPPED=$((MAPPED + 1))
  else
    echo "  WARNING: Could not auto-map folder '$FOLDER' — skipping."
    echo "           Manually copy its images to one of:"
    echo "             $DATA_DIR/cardiovascular_disease/"
    echo "             $DATA_DIR/cardiomyopathy/"
    echo "             $DATA_DIR/congenital_heart_disease/"
    echo "             $DATA_DIR/normal/"
  fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Final class counts ==="
for CLASS in cardiovascular_disease cardiomyopathy congenital_heart_disease normal; do
  COUNT=$(find "$DATA_DIR/$CLASS" -type f | wc -l | tr -d ' ')
  echo "  $CLASS: $COUNT images"
done

echo ""
if [ $MAPPED -eq 0 ]; then
  echo "WARNING: No folders were auto-mapped."
  echo "Run:  ls $TMP_DIR"
  echo "Then manually copy images into the class folders under data/heart_mri/"
else
  echo "Dataset ready at: $DATA_DIR"
  echo ""
  echo "Next step — activate the venv and start training:"
  echo "  source venv/bin/activate"
  echo "  python main.py train --skip-registration"
fi

# Clean up temp download folder
rm -rf "$TMP_DIR"
