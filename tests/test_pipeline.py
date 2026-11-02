"""
test_pipeline.py
----------------
Unit tests for the Heart MRI Classification pipeline.

Four test cases, one per pipeline stage:

    TC-01  Dataset gathering and stratified splitting
    TC-02  Image preprocessing pipeline
    TC-03  Feature extraction (HOG + ResNet-50 deep features)
    TC-04  Model loading and inference

Run:
    python -m unittest discover -s tests -v
"""

# Threading guard — must precede torch / skimage imports (matches main.py).
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLASSES, DATASET_ROOT, DEVICE, IMAGE_SIZE, MODEL_DIR,
    NUM_CLASSES, RANDOM_SEED, TRAIN_SPLIT,
)


def _find_sample_image() -> Path:
    """Return one real MRI image from the dataset for use as test input."""
    for cls in CLASSES:
        cls_dir = DATASET_ROOT / cls
        if cls_dir.is_dir():
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    return p
    raise unittest.SkipTest("No dataset images available")


# ===========================================================================
# TEST CASE 1 — Dataset gathering and stratified splitting
# ===========================================================================

class TC01_DatasetGathering(unittest.TestCase):
    """Verify images are discovered correctly and split 80/20 without leakage."""

    @classmethod
    def setUpClass(cls):
        from src.data_loader import discover_dataset, split_dataset
        cls.dataset = discover_dataset(DATASET_ROOT)
        cls.train, cls.test = split_dataset(cls.dataset, TRAIN_SPLIT, RANDOM_SEED)

    def test_01a_classes_discovered(self):
        """Discovered class names must be a subset of the declared taxonomy."""
        self.assertTrue(len(self.dataset) > 0, "No classes discovered")
        for cls_name in self.dataset:
            self.assertIn(cls_name, CLASSES)
        print(f"\n    → classes found: {list(self.dataset.keys())}")

    def test_01b_all_paths_exist(self):
        """Every discovered path must resolve to a readable file."""
        for cls_name, paths in self.dataset.items():
            self.assertGreater(len(paths), 0, f"{cls_name} is empty")
            for p in paths[:50]:                      # sample check
                self.assertTrue(p.exists(), f"Missing file: {p}")

    def test_01c_split_ratio_is_80_20(self):
        """Train split must hold ~80% of all samples (±1%)."""
        total = len(self.train) + len(self.test)
        ratio = len(self.train) / total
        print(f"    → train={len(self.train)}  test={len(self.test)}  "
              f"ratio={ratio:.4f}")
        self.assertAlmostEqual(ratio, TRAIN_SPLIT, delta=0.01)

    def test_01d_no_overlap_between_splits(self):
        """No image may appear in both train and test (data leakage check)."""
        train_paths = {p for p, _ in self.train}
        test_paths = {p for p, _ in self.test}
        self.assertEqual(len(train_paths & test_paths), 0,
                         "Leakage: image present in both splits")

    def test_01e_split_is_stratified(self):
        """Each class must keep its 80/20 proportion inside the split."""
        for cls_name, paths in self.dataset.items():
            idx = CLASSES.index(cls_name)
            n_train = sum(1 for _, y in self.train if y == idx)
            self.assertAlmostEqual(n_train / len(paths), TRAIN_SPLIT, delta=0.01)

    def test_01f_split_is_reproducible(self):
        """The same seed must always produce an identical split."""
        from src.data_loader import split_dataset
        train2, _ = split_dataset(self.dataset, TRAIN_SPLIT, RANDOM_SEED)
        self.assertEqual([p for p, _ in self.train][:100],
                         [p for p, _ in train2][:100])


# ===========================================================================
# TEST CASE 2 — Preprocessing pipeline
# ===========================================================================

class TC02_Preprocessing(unittest.TestCase):
    """Verify each preprocessing stage produces a valid normalised image."""

    @classmethod
    def setUpClass(cls):
        from src.preprocessing import preprocess_image
        cls.sample = _find_sample_image()
        cls.result = preprocess_image(cls.sample, skip_registration=True)

    def test_02a_output_shape(self):
        """Output must be resized to the model input size (224×224)."""
        print(f"\n    → input : {self.sample.name}")
        print(f"    → output shape: {self.result.shape}")
        self.assertEqual(self.result.shape, IMAGE_SIZE)

    def test_02b_output_dtype_float32(self):
        """Output must be float32 for downstream torch compatibility."""
        self.assertEqual(self.result.dtype, np.float32)

    def test_02c_intensity_range_normalised(self):
        """All pixel intensities must lie in [0, 1]."""
        print(f"    → range: [{self.result.min():.4f}, {self.result.max():.4f}]")
        self.assertGreaterEqual(self.result.min(), 0.0)
        self.assertLessEqual(self.result.max(), 1.0)

    def test_02d_output_not_blank(self):
        """Preprocessing must not destroy image content (non-zero variance)."""
        self.assertGreater(float(self.result.std()), 0.01,
                           "Preprocessed image is flat/blank")

    def test_02e_intensity_standardise_rescales(self):
        """Percentile clip + rescale must map any input onto [0, 1]."""
        from src.preprocessing import intensity_standardise
        noisy = np.random.rand(64, 64).astype(np.float32) * 500 - 100
        out = intensity_standardise(noisy)
        self.assertAlmostEqual(float(out.min()), 0.0, places=4)
        self.assertAlmostEqual(float(out.max()), 1.0, places=4)

    def test_02f_deterministic(self):
        """Same input must always yield the same preprocessed output."""
        from src.preprocessing import preprocess_image
        again = preprocess_image(self.sample, skip_registration=True)
        np.testing.assert_allclose(self.result, again, rtol=1e-5)


# ===========================================================================
# TEST CASE 3 — Feature extraction
# ===========================================================================

class TC03_FeatureExtraction(unittest.TestCase):
    """Verify HOG and deep feature extractors produce correct dimensions."""

    @classmethod
    def setUpClass(cls):
        from src.preprocessing import preprocess_image
        cls.sample = _find_sample_image()
        cls.image = preprocess_image(cls.sample, skip_registration=True)

    def test_03a_hog_vector_is_1d_float32(self):
        """HOG descriptor must be a 1-D float32 vector."""
        from src.feature_extraction import extract_hog_features
        fd = extract_hog_features(self.image)
        print(f"\n    → HOG vector shape: {fd.shape}")
        self.assertEqual(fd.ndim, 1)
        self.assertEqual(fd.dtype, np.float32)
        self.assertGreater(fd.size, 0)

    def test_03b_hog_length_is_consistent(self):
        """HOG length must be identical for every 224×224 input."""
        from src.feature_extraction import extract_hog_features
        a = extract_hog_features(self.image)
        b = extract_hog_features(np.random.rand(*IMAGE_SIZE).astype(np.float32))
        self.assertEqual(a.shape, b.shape)

    def test_03c_hog_is_informative(self):
        """HOG descriptor must not be an all-zero vector."""
        from src.feature_extraction import extract_hog_features
        fd = extract_hog_features(self.image)
        self.assertGreater(float(np.abs(fd).sum()), 0.0)

    def test_03d_deep_features_are_2048d(self):
        """ResNet-50 embedding must be 2048-dimensional per image."""
        from src.feature_extraction import ResNetFeatureExtractor
        from src.config import RESNET_FEATURE_DIM
        extractor = ResNetFeatureExtractor().to(DEVICE).eval()
        batch = torch.randn(2, 3, *IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            feats = extractor(batch)
        print(f"    → deep feature shape: {tuple(feats.shape)}")
        self.assertEqual(tuple(feats.shape), (2, RESNET_FEATURE_DIM))

    def test_03e_saved_feature_files_align(self):
        """Saved X/y .npy pairs must have matching sample counts."""
        from src.config import FEATURES_DIR
        for base in ("train_hog", "test_hog", "train_deep", "test_deep"):
            xp, yp = FEATURES_DIR / f"{base}_X.npy", FEATURES_DIR / f"{base}_y.npy"
            if not (xp.exists() and yp.exists()):
                continue
            X, y = np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")
            self.assertEqual(X.shape[0], y.shape[0], f"{base}: X/y length mismatch")
            print(f"    → {base}: X={X.shape}  y={y.shape}")


# ===========================================================================
# TEST CASE 4 — Model loading and inference
# ===========================================================================

class TC04_ModelInference(unittest.TestCase):
    """Verify the trained checkpoint loads and returns valid predictions."""

    @classmethod
    def setUpClass(cls):
        from src.model import load_model
        cls.checkpoint = MODEL_DIR / "best_model.pth"
        if not cls.checkpoint.exists():
            raise unittest.SkipTest(f"Checkpoint not found: {cls.checkpoint}")
        cls.model = load_model(cls.checkpoint)

    def test_04a_checkpoint_loads(self):
        """Saved checkpoint must load into the model without error."""
        print(f"\n    → checkpoint: {self.checkpoint.name}")
        self.assertIsNotNone(self.model)

    def test_04b_model_in_eval_mode(self):
        """Loaded model must be in eval mode (dropout/BN disabled)."""
        self.assertFalse(self.model.training)

    def test_04c_output_shape_matches_num_classes(self):
        """Forward pass must return one logit per class."""
        batch = torch.randn(4, 3, *IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            logits = self.model(batch)
        print(f"    → logits shape: {tuple(logits.shape)}")
        self.assertEqual(tuple(logits.shape), (4, NUM_CLASSES))

    def test_04d_probabilities_sum_to_one(self):
        """Softmax output must form a valid probability distribution."""
        batch = torch.randn(4, 3, *IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            proba = torch.softmax(self.model(batch), dim=1)
        sums = proba.sum(dim=1).cpu().numpy()
        np.testing.assert_allclose(sums, np.ones(4), rtol=1e-4)
        self.assertTrue((proba >= 0).all() and (proba <= 1).all())

    def test_04e_prediction_on_real_image(self):
        """End-to-end: a real MRI image must yield a valid class prediction."""
        from src.inference import predict_single
        sample = _find_sample_image()
        result = predict_single(str(sample), model=self.model)
        print(f"    → predicted: {result['class']}  "
              f"(confidence {result['confidence']:.4f})")
        self.assertIn(result["class"], CLASSES)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_04f_inference_is_deterministic(self):
        """Same input must produce the same logits on repeated calls."""
        batch = torch.randn(1, 3, *IMAGE_SIZE).to(DEVICE)
        with torch.no_grad():
            a = self.model(batch)
            b = self.model(batch)
        torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
