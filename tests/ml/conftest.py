"""Fixtures for the ML pipeline tests.

The classification pipeline is built around dependency injection (classifier,
writer, best-frame selector, record/video callables are all passed in), so these
lightweight fakes exercise the real orchestration without a camera, an ONNX model,
or a database.

For the end-to-end tests that DO need the real model, this module also exposes
``bird_image_cases`` (the labelled ``tests/_test_images`` fixtures, cropped from
their hand-annotated boxes) and ``real_classifier`` (the int8 ONNX classifier).
Both skip cleanly when their out-of-band files (the JPEGs / the ONNX model) are
absent, matching the rest of the model-dependent suite.
"""

import json
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

import numpy as np
import pytest
from PIL import Image

from birdscanner.ml.classification import Classifier
from birdscanner.ml.classification_pipeline import setup_classifier

# This file lives at tests/ml/, so the repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
BBOX_MANIFEST = _REPO_ROOT / "tests" / "_test_images" / "bounding_box_locations.json"
INT8_MODEL_PATH = _REPO_ROOT / "assets" / "models" / "convnext_v2_tiny_int8.onnx"
CLASS_TO_IDX_PATH = (
    _REPO_ROOT / "assets" / "models" / "convnext_v2_tiny.onnx_class_to_idx.json"
)


class ImageCase(NamedTuple):
    """One labelled test image: the frame, its bird box, and expected species.

    Attributes:
        name: The image filename (for assertion messages).
        image: Full-resolution RGB frame as an ``(H, W, 3)`` uint8 array.
        box: The bird's bounding box as ``(x, y, w, h)`` in frame pixels.
        species: The expected classifier label for the bird.
    """

    name: str
    image: np.ndarray
    box: Tuple[int, int, int, int]
    species: str


class FakeDetection:
    """Stand-in for ``object_detection.Detection`` with the fields the pipeline reads."""

    def __init__(
        self, box: Tuple[int, int, int, int], conf: float = 0.9, category: int = 0
    ) -> None:
        """Record the box, detection confidence, and category index."""
        self.box = box
        self.conf = conf
        self.category = category


class RecordingWriter:
    """``DetectionWriter`` stand-in that captures ``write()`` calls instead of persisting."""

    def __init__(self) -> None:
        """Start with an empty capture log."""
        self.writes: List[Any] = []

    def write(self, record: Any) -> None:
        """Record one written ``DetectionRecord``."""
        self.writes.append(record)


class RecordingRecorder:
    """Records ``record_fn`` invocations and returns a preset started/declined result."""

    def __init__(self, started: bool = True) -> None:
        """Configure whether the fake recorder reports that recording began."""
        self.started = started
        self.paths: List[str] = []

    def __call__(self, path: str) -> bool:
        """Log the requested path and report whether recording started."""
        self.paths.append(path)
        return self.started


@pytest.fixture()
def fake_detection() -> type[FakeDetection]:
    """Expose the :class:`FakeDetection` class."""
    return FakeDetection


@pytest.fixture()
def recording_writer() -> RecordingWriter:
    """A fresh :class:`RecordingWriter`."""
    return RecordingWriter()


@pytest.fixture()
def recording_recorder() -> "type[RecordingRecorder]":
    """Expose the :class:`RecordingRecorder` class so tests choose started/declined."""
    return RecordingRecorder


@pytest.fixture()
def stable_tracker():
    """Return a builder for a tracker with one immediately-stable detection.

    Calling the returned function with a detection runs a single ``update_frame`` at
    ``min_stable_frames=1``, so ``detection_id`` 0 is stable and maps to a track.
    """
    from birdscanner.ml.tracking import StableDetectionTracker

    def _make(detection: Optional[FakeDetection] = None) -> StableDetectionTracker:
        tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=1)
        if detection is not None:
            tracker.update_frame([detection])
        return tracker

    return _make


def _load_bird_image_cases() -> List[ImageCase]:
    """Load the labelled test images and convert their boxes to ``(x, y, w, h)``.

    The manifest stores each box as ``[[tlx, tly], [brx, bry]]`` (top-left and
    bottom-right corners); the pipeline works in ``(x, y, w, h)``, so the width
    and height are derived here.

    Returns:
        One :class:`ImageCase` per manifest entry.

    Raises:
        pytest.skip.Exception: If the manifest or any referenced image is absent.
    """
    if not BBOX_MANIFEST.exists():
        pytest.skip(f"Bounding-box manifest {BBOX_MANIFEST} not found.")
    entries = json.loads(BBOX_MANIFEST.read_text(encoding="utf-8"))
    cases: List[ImageCase] = []
    for entry in entries:
        image_path = _REPO_ROOT / entry["image"]
        if not image_path.exists():
            pytest.skip(f"Test image {image_path} not found.")
        frame = np.asarray(Image.open(image_path).convert("RGB"))
        (tlx, tly), (brx, bry) = entry["bounding_box"]
        box = (tlx, tly, brx - tlx, bry - tly)
        cases.append(ImageCase(image_path.name, frame, box, entry["species"]))
    return cases


@pytest.fixture(scope="module")
def bird_image_cases() -> List[ImageCase]:
    """Labelled ``tests/_test_images`` frames + boxes + expected species.

    Module-scoped so the (large) JPEGs are decoded once per test module. The
    frames are only ever read (cropped into fresh arrays), never mutated, so
    sharing them across tests is safe.
    """
    return _load_bird_image_cases()


@pytest.fixture(scope="module")
def real_classifier() -> Classifier:
    """The real int8 ONNX species classifier, skipping if the model is absent.

    Module-scoped so the ONNX session is created once per test module; inference
    is stateless, so the session is safe to share across tests.
    """
    if not INT8_MODEL_PATH.exists():
        pytest.skip(f"Classifier model {INT8_MODEL_PATH} not found.")
    return setup_classifier(str(INT8_MODEL_PATH), str(CLASS_TO_IDX_PATH))
