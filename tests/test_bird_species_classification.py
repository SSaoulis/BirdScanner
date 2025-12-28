import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import re

from src.classification import ONNXClassifier, Classifier, build_preprocessing
# add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))  # noqa: E

# Paths
MODEL_PATH = Path(__file__).parent.parent / "examples/models/convnext_v2_tiny.onnx"
CLASS_TO_IDX_PATH = Path(__file__).parent.parent / "src/assets/convnext_v2_tiny.onnx_class_to_idx.json"
IMAGES_DIR = Path(__file__).parent / "bird_species"


# minimal preparation: resize to model expected size 384x384, convert to NCHW float32 (no normalization)
def prepare_input(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((384, 384), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def _normalize_name(name: str) -> str:
    s = name.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = " ".join(s.split())
    return s


def test_bird_species_images_prediction():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file {MODEL_PATH} does not exist. Skipping test.")
    if not CLASS_TO_IDX_PATH.exists():
        pytest.skip(f"Class mapping file {CLASS_TO_IDX_PATH} does not exist. Skipping test.")
    if not IMAGES_DIR.exists():
        pytest.skip(f"Images directory {IMAGES_DIR} does not exist. Skipping test.")

    onnx_model = ONNXClassifier(str(MODEL_PATH))
    classifier = Classifier(onnx_model, CLASS_TO_IDX_PATH, preprocessing=None)

    image_files = sorted(list(IMAGES_DIR.glob("*.jpg"))) + sorted(list(IMAGES_DIR.glob("*.jpeg")))
    if not image_files:
        pytest.skip(f"No JPEG images found in {IMAGES_DIR}. Skipping test.")

    for img_path in image_files:
        img = Image.open(img_path)
        input_tensor = prepare_input(img)
        cls_name, conf = classifier.classify(input_tensor)
        print(f"Image: {img_path.name} -> Prediction: {cls_name} (conf {conf:.4f})")
        # check predicted class roughly matches the filename
        expected = _normalize_name(img_path.stem)
        pred_norm = _normalize_name(cls_name)
        assert expected in pred_norm or pred_norm in expected
        # basic sanity assertions
        assert isinstance(cls_name, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0


def test_bird_species_images_prediction_with_preprocessing():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file {MODEL_PATH} does not exist. Skipping test.")
    if not CLASS_TO_IDX_PATH.exists():
        pytest.skip(f"Class mapping file {CLASS_TO_IDX_PATH} does not exist. Skipping test.")
    if not IMAGES_DIR.exists():
        pytest.skip(f"Images directory {IMAGES_DIR} does not exist. Skipping test.")

    onnx_model = ONNXClassifier(str(MODEL_PATH))
    preprocessing = build_preprocessing({
        "size": (384, 384),
        "rgb_values": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "center_crop": 1.0,
        "simple_crop": False,
    })
    classifier = Classifier(onnx_model, CLASS_TO_IDX_PATH, preprocessing=preprocessing)

    image_files = sorted(list(IMAGES_DIR.glob("*.jpg"))) + sorted(list(IMAGES_DIR.glob("*.jpeg")))
    if not image_files:
        pytest.skip(f"No JPEG images found in {IMAGES_DIR}. Skipping test.")

    for img_path in image_files:
        img = Image.open(img_path)
        cls_name, conf = classifier.classify(img)
        print(f"[preproc] Image: {img_path.name} -> Prediction: {cls_name} (conf {conf:.4f})")
        expected = _normalize_name(img_path.stem)
        pred_norm = _normalize_name(cls_name)
        assert expected in pred_norm or pred_norm in expected
        assert isinstance(cls_name, str)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0
