# create a simple test that uses class ONNXClassifier from src/classification.py

import pytest

from src.classification import ONNXClassifier, Classifier, build_preprocessing
import numpy as np
import os
from pathlib import Path


# REAL MODEL PATH
MODEL_PATH = Path(__file__).parent.parent / "examples/models/convnext_v2_tiny_int8.onnx"
CLASS_TO_IDX_PATH = Path(__file__).parent.parent / "src/assets/convnext_v2_tiny.onnx_class_to_idx.json"

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
    preprocessing = build_preprocessing({
        "size": (384, 384),
        "rgb_values": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "center_crop": 1.0,
        "simple_crop": False,
    })
    classifier = Classifier(onnx_model, CLASS_TO_IDX_PATH, preprocessing=preprocessing)

    # Create a random image-like numpy array (H,W,3) in [0,1]
    rng_img = np.random.rand(384, 384, 3).astype(np.float32)

    # Perform prediction (preprocessing will convert to NCHW)
    output = classifier.predict(rng_img)

    # Check output shape
    assert output.ndim == 2
    assert output.shape[0] == 1
    assert output.shape[1] > 0

# end
# To run the test, use the command: pytest tests/test_classification.py
