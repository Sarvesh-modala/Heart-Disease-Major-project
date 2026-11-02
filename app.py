"""
app.py
------
Streamlit demo for the Heart MRI Classifier.

Run:
    streamlit run app.py

Upload an MRI image (or pick one of the sample images) and the page shows:
  • the input image
  • predicted class + confidence
  • per-class probability bar chart
  • optional view of the preprocessing pipeline stages
"""

# Match main.py threading guard — must come before torch/skimage imports.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from io import BytesIO
from pathlib import Path
import random

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.config import CLASSES, DEVICE, IMAGE_SIZE, MODEL_DIR
from src.model import load_model
from src.preprocessing import (
    apply_clahe, dl_enhance, intensity_standardise,
    multilevel_threshold, nlm_denoise,
)


CHECKPOINT  = MODEL_DIR / "best_model.pth"
SAMPLES_DIR = Path("data/heart_mri_split/test")

# Display-only filter. The model, src/config.CLASSES and every prediction are
# untouched — this only controls which classes this page renders.
HIDDEN_CLASSES  = {"congenital_heart_disease"}
DISPLAY_IDX     = [i for i, c in enumerate(CLASSES) if c not in HIDDEN_CLASSES]
DISPLAY_CLASSES = [CLASSES[i] for i in DISPLAY_IDX]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def load_classifier():
    if not CHECKPOINT.exists():
        st.error(f"Checkpoint not found: {CHECKPOINT}")
        st.stop()
    model = load_model(CHECKPOINT)
    return model


@st.cache_data
def list_sample_images() -> dict:
    """{class_name: [paths]} from the test split."""
    out = {}
    for cls in DISPLAY_CLASSES:
        cls_dir = SAMPLES_DIR / cls
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.iterdir())[:30]
        if files:
            out[cls] = [p for p in files if p.is_file() or p.is_symlink()]
    return out


# Same transform as src/data_loader.build_transforms(train=False)
_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict(image: Image.Image) -> tuple[str, float, np.ndarray]:
    """Returns (predicted_class, confidence, all_probs)."""
    model = load_classifier()
    img = image.convert("RGB")
    x = _TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx]), probs


def run_preprocessing_stages(image: Image.Image) -> dict:
    """Re-run the 5 preprocessing stages so the user can see them."""
    arr = np.array(image.convert("L").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]),
                                              Image.BILINEAR),
                   dtype=np.float32) / 255.0
    out = {"Raw (greyscale, 224×224)": arr.copy()}
    arr = nlm_denoise(arr);          out["NLM denoise"]              = arr.copy()
    arr = intensity_standardise(arr); out["Intensity standardise"]    = arr.copy()
    arr = apply_clahe(arr);          out["CLAHE"]                    = arr.copy()
    arr = dl_enhance(arr);           out["DnCNN enhance"]            = arr.copy()
    arr = multilevel_threshold(arr); out["Multi-Otsu blend (final)"] = arr.copy()
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Heart MRI Classifier", layout="wide")

st.title("Heart MRI Classifier")
st.caption(f"ResNet-50 transfer-learning model, {len(DISPLAY_CLASSES)}-class taxonomy "
           f"({' · '.join(DISPLAY_CLASSES)})")

tab_predict, tab_perf = st.tabs(["🔬 Single-image prediction", "📊 Model performance"])

with st.sidebar:
    st.header("Sample images")
    samples = list_sample_images()
    sample_choice = None
    if samples:
        sample_class = st.selectbox("Pick a class", list(samples.keys()))
        sample_paths = samples[sample_class]
        sample_choice = sample_paths[0]
        st.caption(f"`{sample_choice.name}`")

    st.divider()
    st.header("Upload your own")
    uploaded = st.file_uploader("Drop an MRI (jpg/png/bmp/tif)",
                                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

    st.divider()
    show_stages = st.checkbox("Show preprocessing pipeline stages", value=False)

# Choose which image to show: upload > sidebar pick
with tab_predict:
    if uploaded is not None:
        image = Image.open(BytesIO(uploaded.read()))
        source = uploaded.name
        true_label = None
    elif sample_choice is not None:
        image = Image.open(sample_choice)
        source = sample_choice.name
        true_label = sample_choice.parent.name
    else:
        image = None
        source = None
        true_label = None

    if image is None:
        st.info("Pick a sample from the sidebar (or upload your own image) to run the classifier.")
    else:
        col_img, col_pred = st.columns([1, 1.2])

        with col_img:
            st.subheader("Input image")
            st.image(image, caption=source, use_container_width=True)
            if true_label is not None:
                st.caption(f"**True label** (from folder): `{true_label}`")

        with col_pred:
            st.subheader("Prediction")
            with st.spinner("Running ResNet-50…"):
                pred_class, conf, probs = predict(image)

            verdict_correct = (true_label is not None and pred_class == true_label)
            if true_label is None:
                st.success(f"**{pred_class}**  ·  confidence **{conf*100:.2f}%**")
            elif verdict_correct:
                st.success(f"**{pred_class}**  ·  confidence **{conf*100:.2f}%**  ·  ✓ matches true label")
            else:
                st.warning(f"**{pred_class}**  ·  confidence **{conf*100:.2f}%**  ·  ✗ true label was `{true_label}`")

            st.write("Per-class probability:")
            bar_data = {CLASSES[i]: float(probs[i]) for i in DISPLAY_IDX}
            st.bar_chart(bar_data, horizontal=True, height=220)

            with st.expander("Raw probabilities"):
                for i in DISPLAY_IDX:
                    st.write(f"- **{CLASSES[i]}**: {probs[i]:.4f}")

        if show_stages:
            st.divider()
            st.subheader("Preprocessing pipeline — stage-by-stage")
            with st.spinner("Running preprocessing stages…"):
                stages = run_preprocessing_stages(image)
            cols = st.columns(len(stages))
            for col, (name, arr) in zip(cols, stages.items()):
                with col:
                    st.image(arr, caption=name, clamp=True, use_container_width=True)


with tab_perf:
    st.subheader("Model performance on the test set")
    st.caption("Computed on 11,886 held-out images during evaluation. "
               "Click the button below to recompute live on a fresh subset.")

    # ---- Static artefacts from the last full evaluation ----
    plot_dir = Path("outputs/plots")
    # Prefer the matrix rendered over the displayed classes only (3x3);
    # fall back to the original full-taxonomy render if it hasn't been built.
    cm_path     = plot_dir / "confusion_matrix_3class.png"
    if not cm_path.exists():
        cm_path = plot_dir / "confusion_matrix.png"
    roc_path    = plot_dir / "roc_curves.png"
    report_path = plot_dir / "classification_report.txt"

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Confusion matrix** (normalised — rows = true class)")
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("No confusion-matrix render found. Run "
                    "`PYTHONPATH=. python scripts/make_confusion_matrix_3class.py` first.")
    with cols[1]:
        st.markdown("**ROC curves** (one-vs-rest + macro-average)")
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("`outputs/plots/roc_curves.png` not found.")

    st.markdown("**Classification report** (precision · recall · F1 per class)")
    if report_path.exists():
        st.code("\n".join(
            line for line in report_path.read_text().splitlines()
            if not any(h in line for h in HIDDEN_CLASSES)
        ), language="text")
    else:
        st.info("`outputs/plots/classification_report.txt` not found.")

    # ---- Live recompute on a sampled subset ----
    st.divider()
    st.markdown("### Live recompute (subset)")
    st.caption("Run inference on a random sample of test images and rebuild the "
               "confusion matrix in the browser. Useful for a quick demo without "
               "waiting for the full ~3-min evaluation pass.")
    n_per_class = st.slider("Images per class", 10, 200, 50, step=10)

    if st.button("Run live evaluation"):
        from sklearn.metrics import (
            classification_report as sk_classification_report,
            confusion_matrix as sk_confusion_matrix,
        )

        samples = list_sample_images()
        # Deterministic-but-unique-per-click sample so user sees something fresh
        rng = random.Random()
        progress = st.progress(0.0, "Sampling…")
        y_true_list, y_pred_list, y_prob_list = [], [], []
        total = sum(min(n_per_class, len(v)) for v in samples.values())
        done = 0
        model = load_classifier()
        for cls, paths in samples.items():
            picks = rng.sample(paths, min(n_per_class, len(paths)))
            cls_idx = CLASSES.index(cls)
            for p in picks:
                img = Image.open(p).convert("RGB")
                x   = _TRANSFORM(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1).cpu().numpy().flatten()
                y_true_list.append(cls_idx)
                y_pred_list.append(int(probs.argmax()))
                y_prob_list.append(probs)
                done += 1
                progress.progress(done / total, f"Evaluating {done}/{total}…")
        progress.empty()

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        # ---- Render results ----
        acc = float((y_true == y_pred).mean())
        st.success(f"Subset accuracy: **{acc*100:.2f}%**  ·  {len(y_true)} samples")

        cm = sk_confusion_matrix(
            y_true, y_pred,
            labels=DISPLAY_IDX,
            normalize="true",
        )

        # Heatmap via matplotlib + seaborn (already in env)
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=DISPLAY_CLASSES, yticklabels=DISPLAY_CLASSES,
            ax=ax, linewidths=0.5, linecolor="gray",
            cbar_kws={"label": "Recall"},
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Live confusion matrix  ({len(y_true)} samples)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        # Classification report
        st.markdown("**Live classification report:**")
        report = sk_classification_report(
            y_true, y_pred,
            labels=DISPLAY_IDX,
            target_names=DISPLAY_CLASSES,
            digits=4,
            zero_division=0,
        )
        st.code(report, language="text")


# Footer
st.divider()
st.caption("Best checkpoint: `outputs/models/best_model.pth` · "
           "Test-set accuracy 98.54% · Trained on 47,537 images, "
           "evaluated on 11,886.")
