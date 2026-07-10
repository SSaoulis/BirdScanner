"""Tests for the off-Pi YOLO ONNX object detector.

The postprocessing (letterbox, decode, NMS) is tested with synthetic tensors so
no model is needed.  A separate real-model test runs the actual ONNX against a
bundled bird JPEG and is skipped when the model file is absent (matching the
project's out-of-band-model convention).
"""

from pathlib import Path

import numpy as np
import pytest

from birdscanner.detector.emulation.yolo import (
    COCO_CLASSES,
    Detected,
    OnnxYoloDetector,
    _letterbox,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_YOLO_MODEL = _REPO_ROOT / "assets" / "models" / "yolo11n.onnx"
_BIRD_IMAGE = _REPO_ROOT / "tests" / "_test_images" / "Erithacus_rubecula.jpg"
_BIRD_INDEX = COCO_CLASSES.index("bird")


def test_letterbox_preserves_aspect_and_pads():
    """A non-square frame is scaled by one ratio and padded to a square canvas."""
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)  # 2:1 landscape
    canvas, transform = _letterbox(frame, 640)

    assert canvas.shape == (640, 640, 3)
    assert transform.ratio == pytest.approx(3.2)  # 640 / 200 (the longer side)
    # Width fills the canvas; height is padded top and bottom.
    assert transform.pad_x == 0
    assert transform.pad_y == (640 - int(round(100 * transform.ratio))) // 2
    assert (transform.width, transform.height) == (200, 100)


def test_letterbox_transform_maps_box_back_to_normalized():
    """to_normalized undoes the padding/scale into [0, 1] frame fractions."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    _, transform = _letterbox(frame, 640)
    # A centre box of the letterboxed canvas maps to the frame centre.
    cx = cy = 320.0
    x0, y0, x1, y1 = transform.to_normalized(np.array([cx, cy, 64.0, 64.0]))
    assert 0.0 <= x0 < x1 <= 1.0
    assert 0.0 <= y0 < y1 <= 1.0
    assert (x0 + x1) / 2 == pytest.approx(0.5, abs=0.02)
    assert (y0 + y1) / 2 == pytest.approx(0.5, abs=0.02)


def _detector_without_session(size=640, conf=0.35, iou=0.45) -> OnnxYoloDetector:
    """Build an OnnxYoloDetector bypassing __init__ (no ONNX session needed)."""
    detector = OnnxYoloDetector.__new__(OnnxYoloDetector)
    detector._size = size  # pylint: disable=protected-access
    detector._conf_threshold = conf  # pylint: disable=protected-access
    detector._iou_threshold = iou  # pylint: disable=protected-access
    return detector


def test_decode_filters_by_confidence_and_picks_class():
    """_decode extracts boxes/scores and argmax class above the threshold."""
    detector = _detector_without_session()
    outputs = np.zeros((1, 84, 8400), dtype=np.float32)
    # Anchor 0: a confident bird; anchor 1: a low-confidence cat (dropped).
    outputs[0, :4, 0] = [100.0, 120.0, 40.0, 30.0]
    outputs[0, 4 + _BIRD_INDEX, 0] = 0.9
    outputs[0, :4, 1] = [10.0, 10.0, 5.0, 5.0]
    outputs[0, 4 + COCO_CLASSES.index("cat"), 1] = 0.1

    boxes, scores, class_ids = detector._decode(
        outputs
    )  # pylint: disable=protected-access

    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(boxes[0], [100.0, 120.0, 40.0, 30.0])
    assert scores[0] == pytest.approx(0.9)
    assert int(class_ids[0]) == _BIRD_INDEX


def test_nms_keeps_highest_of_overlapping_boxes():
    """Overlapping boxes are suppressed down to the highest-scoring one."""
    detector = _detector_without_session(iou=0.5)
    # Two near-identical centre boxes + one far away.
    boxes = np.array(
        [[100, 100, 40, 40], [102, 101, 40, 40], [400, 400, 20, 20]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    keep = set(detector._nms(boxes, scores))  # pylint: disable=protected-access

    assert 0 in keep  # the highest of the overlapping pair survives
    assert 1 not in keep  # its overlapping neighbour is suppressed
    assert 2 in keep  # the distant box is independent


def test_nms_empty_input_returns_empty():
    """NMS on no boxes returns no indices."""
    detector = _detector_without_session()
    assert (
        detector._nms(np.empty((0, 4)), np.empty((0,))) == []
    )  # pylint: disable=protected-access


@pytest.mark.skipif(
    not (_YOLO_MODEL.exists() and _BIRD_IMAGE.exists()),
    reason="yolo11n.onnx or bird test image absent",
)
def test_real_detector_runs_on_bird_image():
    """The real ONNX detector runs end to end and returns valid detections."""
    from PIL import Image

    detector = OnnxYoloDetector(str(_YOLO_MODEL))
    frame = np.asarray(Image.open(_BIRD_IMAGE).convert("RGB"))
    detections = detector.detect(frame)

    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, Detected)
        x0, y0, x1, y1 = det.box
        assert 0.0 <= x0 <= x1 <= 1.0
        assert 0.0 <= y0 <= y1 <= 1.0
        assert 0.0 <= det.score <= 1.0
    # The fixture is a robin filling much of the frame — expect a bird among them.
    assert any(det.label == "bird" for det in detections)
