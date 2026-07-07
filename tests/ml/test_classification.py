# create a simple test that uses class ONNXClassifier from birdscanner/ml/classification.py

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from birdscanner.ml.classification import (
    Classifier,
    ONNXClassifier,
    build_preprocessing,
)

# REAL MODEL PATH (this file lives at tests/ml/, so the repo root is two parents up).
_REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _REPO_ROOT / "assets/models/convnext_v2_tiny_int8.onnx"
CLASS_TO_IDX_PATH = _REPO_ROOT / "assets/models/convnext_v2_tiny.onnx_class_to_idx.json"


# test a simple prediction with the real model and random data of size 384x384x3
def test_onnx_classifier_prediction():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file {MODEL_PATH} does not exist. Skipping test.")

    classifier = ONNXClassifier(str(MODEL_PATH))

    # Create random input data with shape (1, 3, 384, 384)
    input_data = np.random.rand(1, 3, 384, 384).astype(np.float32)

    # Perform prediction
    output = classifier.predict(input_data)

    # Check output shape (should be (1, num_classes), num_classes depends on the model)
    assert output.ndim == 2
    assert output.shape[0] == 1
    assert output.shape[1] > 0  # num_classes should be greater than 0


def test_onnx_classifier_prediction_with_preprocessing():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file {MODEL_PATH} does not exist. Skipping test.")

    # Build model and preprocessing
    onnx_model = ONNXClassifier(str(MODEL_PATH))
    preprocessing = build_preprocessing(
        {
            "size": (384, 384),
            "rgb_values": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "center_crop": 1.0,
            "simple_crop": False,
        }
    )
    classifier = Classifier(onnx_model, CLASS_TO_IDX_PATH, preprocessing=preprocessing)

    # Create a random image-like numpy array (H,W,3) in [0,1]
    rng_img = np.random.rand(384, 384, 3).astype(np.float32)

    # Perform prediction (preprocessing will convert to NCHW)
    output = classifier.predict(rng_img)

    # Check output shape
    assert output.ndim == 2
    assert output.shape[0] == 1
    assert output.shape[1] > 0


# ---------------------------------------------------------------------------
# build_preprocessing — pure PIL/NumPy, needs no ONNX model
# ---------------------------------------------------------------------------


def _preprocessing(size=(384, 384)):
    """Build the standard ImageNet preprocessing callable at the given size."""
    return build_preprocessing(
        {
            "size": size,
            "rgb_values": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "center_crop": 1.0,
            "simple_crop": False,
        }
    )


def test_build_preprocessing_outputs_nchw_float32():
    """Preprocessing resizes any input to a (1, 3, H, W) float32 batch."""
    pre = _preprocessing((384, 384))
    out = pre(np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8))
    assert out.shape == (1, 3, 384, 384)
    assert out.dtype == np.float32


def test_build_preprocessing_applies_imagenet_normalization():
    """A uniform-grey image normalizes to the expected per-channel value."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pre = build_preprocessing(
        {
            "size": (8, 8),
            "rgb_values": {"mean": mean, "std": std},
            "center_crop": 1.0,
            "simple_crop": False,
        }
    )
    out = pre(np.full((8, 8, 3), 128, dtype=np.uint8))
    grey = 128 / 255.0
    for channel in range(3):
        expected = (grey - mean[channel]) / std[channel]
        assert out[0, channel].mean() == pytest.approx(expected, abs=1e-3)


def test_build_preprocessing_accepts_pil_image():
    """A PIL image input is accepted and produces the configured output size."""
    img = Image.new("RGB", (20, 20), (10, 20, 30))
    out = _preprocessing((16, 16))(img)
    assert out.shape == (1, 3, 16, 16)
