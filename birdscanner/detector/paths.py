"""Package-anchored resolution of the on-disk data files the detector loads.

Paths are resolved relative to this package's location (``<repo>/assets`` next to the
``birdscanner`` package, ``/app/assets`` inside the Docker image), **not** the current
working directory. This lets the detector run as ``python -m birdscanner.detector.main``
from any directory without the old ``cwd=src`` requirement.

Both roots are overridable via environment variables so a deployment can point at a
mounted data volume:

- ``ASSETS_DIR`` — the consolidated asset root (defaults to ``<repo>/assets``).
- ``MODEL_DIR`` — where the ONNX classifier + its class map live (defaults to
  ``ASSETS_DIR/models``).
"""

import os
from pathlib import Path

# ``__file__`` is ``<root>/birdscanner/detector/paths.py``; parents[2] is ``<root>``.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def assets_dir() -> Path:
    """Return the consolidated asset root (``ASSETS_DIR`` env override or ``<repo>/assets``)."""
    override = os.environ.get("ASSETS_DIR")
    return Path(override) if override else _REPO_ROOT / "assets"


def model_dir() -> Path:
    """Return the classifier-model directory (``MODEL_DIR`` env override or ``assets/models``)."""
    override = os.environ.get("MODEL_DIR")
    return Path(override) if override else assets_dir() / "models"


def coco_labels_path() -> Path:
    """Return the path to the COCO label list used to seed the detector intrinsics."""
    return assets_dir() / "labels" / "coco_labels.txt"


def class_to_idx_path() -> Path:
    """Return the path to the ConvNeXt classifier's ``{class_name: index}`` JSON map."""
    return model_dir() / "convnext_v2_tiny.onnx_class_to_idx.json"


def classifier_model_path() -> Path:
    """Return the path to the quantized ConvNeXt V2 Tiny ONNX classifier model."""
    return model_dir() / "convnext_v2_tiny_int8.onnx"
