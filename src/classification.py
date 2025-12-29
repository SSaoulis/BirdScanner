import onnxruntime as ort
import numpy as np
from typing import Callable, Dict, Tuple, Any
from PIL import Image
import json
from pathlib import Path

class ONNXClassifier:
    """
    Lightweight ONNX Runtime classifier wrapper.

    Expected input data
    - Preprocessed batch as a numpy.ndarray with shape (N, C, H, W), dtype float32 (NCHW).
    - Channel order must be RGB, values already normalized/rescaled as required by the model.
    - No preprocessing is performed here; pass data that already matches the model's input size, e.g. (1, 3, 384, 384).
    """
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)

    def predict(self, data):
        """
        Run inference on a pre-batched input.

        Expected input
        - data: numpy.ndarray with shape (N, C, H, W) and dtype float32 (NCHW).
        - Values must already be preprocessed (resize/crop/normalize) to match the model's input contract.

        Returns
        - numpy.ndarray with model outputs. For classifiers this is typically (N, num_classes).
        """
        inputs = {self.session.get_inputs()[0].name: data.astype(np.float32)}
        outputs = self.session.run(None, inputs)
        return outputs[0]
    
def build_classifier(path_to_onnx_model):
    return ONNXClassifier(path_to_onnx_model)

class Classifier:
    """
    Higher-level wrapper that can apply optional preprocessing before ONNX inference.

    Expected input data
    - If preprocessing is None: same as ONNXClassifier, a numpy.ndarray of shape (N, C, H, W), float32.
    - If preprocessing is set: a raw image or array that the preprocessing callable accepts, for example
      a PIL.Image.Image or numpy array shaped (H, W, 3/4) uint8 or float in [0, 1]. The preprocessing
      callable must return a numpy.ndarray (N, C, H, W) float32 batch.

    Parameters
    - model: an object exposing predict(NCHW_float32) -> np.ndarray
    - class_index_path: optional path to a JSON mapping of {class_name: index}. If provided, it enables classify().
    - preprocessing: optional callable to convert raw inputs to NCHW float32.
    """
    def __init__(self, model, class_index_path: str | Path | None = None, preprocessing: Callable[[Any], np.ndarray] | None = None):
        self.model = model
        self.preprocessing = preprocessing
        self.softmax = True
        self.idx_to_class: dict[int, str] | None = None
        if class_index_path is not None:
            p = Path(class_index_path)
            with p.open("r") as f:
                class_to_idx = json.load(f)
            self.idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    def predict(self, data):
        """
        Run inference on a pre-batched input.

        Expected input
        - data: numpy.ndarray with shape (N, C, H, W) and dtype float32 (NCHW).
        - Values must already be preprocessed (resize/crop/normalize) to match the model's input contract.

        Returns
        - numpy.ndarray with model outputs. For classifiers this is typically (N, num_classes).
        """
        # apply preprocessing if provided
        if self.preprocessing is not None:
            data = self.preprocessing(data)
        predictions = self.model.predict(data)
        if self.softmax:
            # apply numerically stable softmax to predictions
            max_logits = np.max(predictions, axis=1, keepdims=True)
            exp_preds = np.exp(predictions - max_logits)
            predictions = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        return predictions

    def classify(self, data) -> tuple[str, float]:
        """
        Classify a single sample and return the predicted class name and confidence.

        Expected input
        - Same as predict(): either preprocessed NCHW float32 or raw input if a preprocessing callable was provided.

        Returns
        - (class_name, confidence) where confidence is the predicted probability for the top class.
        """
        if self.idx_to_class is None:
            raise ValueError("No class index mapping provided. Initialize Classifier with class_index_path to use classify().")
        probs = self.predict(data)
        # handle batch, pick first sample
        top_idx = int(np.argmax(probs, axis=1)[0])
        top_conf = float(probs[0, top_idx])
        class_name = self.idx_to_class.get(top_idx, f"<unknown:{top_idx}>")
        return class_name, top_conf

# data preprocessing (pure PIL + numpy, no torch/torchvision)

# Build a preprocessing callable from a configuration dict.
# Supported keys:
# - size: tuple[int, int]
# - rgb_values: {"mean": list[float], "std": list[float]}
# - center_crop: float (default 1.0)
# - simple_crop: bool (default False)
# The returned callable accepts a PIL.Image.Image or numpy array (H,W,3/4) and returns numpy array (1,3,H,W) float32.

def build_preprocessing(config: Dict[str, Any]) -> Callable[[Any], np.ndarray]:
    size: Tuple[int, int] = tuple(config.get("size", (384, 384)))  # (H, W)
    rgb_values = config.get("rgb_values", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    center_crop: float = float(config.get("center_crop", 1.0))
    simple_crop: bool = bool(config.get("simple_crop", False))

    mean = np.array(rgb_values["mean"], dtype=np.float32)
    std = np.array(rgb_values["std"], dtype=np.float32)

    # compute base size (H, W)
    if simple_crop:
        base_h = base_w = int(min(size) / center_crop)
        base_size = (base_w, base_h)  # PIL expects (W, H)
    else:
        base_h, base_w = int(size[0] / center_crop), int(size[1] / center_crop)
        base_size = (base_w, base_h)  # PIL expects (W, H)

    def _to_pil(x: Any) -> Image.Image:
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, np.ndarray):
            if x.ndim == 3 and x.shape[2] in (3, 4):
                if x.dtype != np.uint8:
                    # assume input in [0,1] floats, clip and scale
                    arr = np.clip(x, 0, 1) * 255.0
                    arr = arr.astype(np.uint8)
                else:
                    arr = x
                img = Image.fromarray(arr)
                return img.convert("RGB")
            elif x.ndim == 2:
                # grayscale
                if x.dtype != np.uint8:
                    arr = np.clip(x, 0, 1) * 255.0
                    arr = arr.astype(np.uint8)
                else:
                    arr = x
                img = Image.fromarray(arr)
                return img.convert("RGB")
        raise TypeError("Unsupported input type for preprocessing. Expected PIL.Image or numpy array")

    def center_crop_pil(img: Image.Image, target_hw: Tuple[int, int]) -> Image.Image:
        target_h, target_w = target_hw
        w, h = img.size
        # img.size returns (W,H)
        left = max((w - target_w) // 2, 0)
        top = max((h - target_h) // 2, 0)
        right = left + target_w
        bottom = top + target_h
        return img.crop((left, top, right, bottom))

    def preprocessing_fn(x: Any) -> np.ndarray:
        # to PIL RGB
        img = _to_pil(x)
        # resize with bicubic to base_size
        img = img.resize(base_size, resample=Image.BICUBIC)
        # center crop to final size (W,H ordering for PIL crop helper)
        img = center_crop_pil(img, (size[0], size[1]))
        # convert to numpy HWC float32 in [0,1]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # normalize
        arr = (arr - mean) / std  # broadcasting over channel dimension
        # HWC -> CHW
        chw = np.transpose(arr, (2, 0, 1))
        # add batch dimension: NCHW
        nchw = np.expand_dims(chw, axis=0).astype(np.float32)
        return nchw

    return preprocessing_fn


def setup_classifier(model_path: str, class_to_idx_path: str):
    """Initialize the ONNX classifier with preprocessing."""
    onnx_model = ONNXClassifier(str(model_path))
    preprocessing = build_preprocessing({
        "size": (384, 384),
        "rgb_values": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "center_crop": 1.0,
        "simple_crop": False,
    })
    return Classifier(onnx_model, class_to_idx_path, preprocessing=preprocessing)
