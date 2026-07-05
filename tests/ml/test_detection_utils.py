"""Unit tests for stateless helpers in ``birdscanner.ml.detection_utils``."""

import numpy as np
import pytest
from PIL import Image

from birdscanner.ml.detection_utils import (
    draw_boxes,
    iou,
    label_for_category,
    normalized_box,
    preprocess_roi,
    save_thumbnail,
)


def test_label_for_category_returns_label_for_valid_index():
    """A valid index returns the corresponding label string."""
    labels = ["person", "bird", "cat"]
    assert label_for_category(labels, 1) == "bird"


def test_label_for_category_returns_none_when_index_out_of_range():
    """An index past the end of the list returns None instead of raising."""
    labels = ["person", "bird", "cat"]
    assert label_for_category(labels, 88) is None


def test_label_for_category_returns_none_for_negative_index():
    """A negative index returns None rather than wrapping around the list."""
    labels = ["person", "bird", "cat"]
    assert label_for_category(labels, -1) is None


def test_label_for_category_returns_none_for_empty_labels():
    """An empty label list never yields a label."""
    assert label_for_category([], 0) is None


def test_normalized_box_divides_by_image_dimensions():
    """A pixel box is converted to fractions of the image width/height."""
    # image_shape is (height, width, channels) like a numpy array.
    nx, ny, nw, nh = normalized_box((100, 150, 200, 300), (600, 400, 3))
    assert nx == pytest.approx(0.25)  # 100 / 400
    assert ny == pytest.approx(0.25)  # 150 / 600
    assert nw == pytest.approx(0.5)  # 200 / 400
    assert nh == pytest.approx(0.5)  # 300 / 600


def test_normalized_box_clamps_to_unit_range():
    """A box extending past the frame edge is clamped to [0, 1]."""
    nx, ny, nw, nh = normalized_box((-10, -20, 500, 800), (600, 400))
    assert nx == 0.0
    assert ny == 0.0
    assert nw == 1.0
    assert nh == 1.0


def test_normalized_box_zero_dimensions_returns_zeros():
    """A degenerate image size yields a zero box instead of dividing by zero."""
    assert normalized_box((0, 0, 10, 10), (0, 0, 3)) == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    """Two identical boxes fully overlap."""
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    """Non-overlapping boxes have zero IoU."""
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_partial_overlap_known_value():
    """A half-overlapping pair yields the analytic IoU."""
    # Two 10x10 boxes offset by 5 in x -> intersection 5x10=50, union 200-50=150.
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_iou_zero_area_boxes_return_zero():
    """A degenerate (zero-area) pair never divides by zero."""
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# ---------------------------------------------------------------------------
# preprocess_roi
# ---------------------------------------------------------------------------


def test_preprocess_roi_returns_square_patch():
    """A non-square box is expanded to a square ROI (with 20% padding)."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    roi, coords = preprocess_roi(image, (20, 30, 20, 40))
    _x, _y, w, h = coords
    assert w == h  # square
    assert roi.shape[0] == roi.shape[1] == w
    assert w > 0


def test_preprocess_roi_clamps_to_image_bounds():
    """A box near the edge stays within the image after expansion + clamping."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    roi, (x, y, w, h) = preprocess_roi(image, (90, 90, 20, 20))
    assert x >= 0 and y >= 0
    assert x + w <= 100 and y + h <= 100
    assert roi.shape[0] == h and roi.shape[1] == w


def test_preprocess_roi_zero_area_box_gives_empty_roi():
    """A zero-area box yields an empty ROI (the pipeline uses this to skip)."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    roi, _ = preprocess_roi(image, (10, 10, 0, 0))
    assert roi.size == 0


# ---------------------------------------------------------------------------
# draw_boxes
# ---------------------------------------------------------------------------


def test_draw_boxes_modifies_image_and_preserves_shape(fake_detection):
    """draw_boxes annotates the frame in place, keeping its shape."""
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    detection = fake_detection(box=(5, 5, 20, 20), conf=0.9, category=0)
    out = draw_boxes(image, (5, 5, 20, 20), detection, ["bird"])
    assert out.shape == (60, 60, 3)
    assert out.any()  # some pixels were drawn


def test_draw_boxes_out_of_range_category_uses_placeholder(fake_detection):
    """An out-of-range category falls back to an ``id:<n>`` label without raising."""
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    detection = fake_detection(box=(5, 5, 20, 20), conf=0.5, category=99)
    # Must not raise despite category 99 being outside the single-label list.
    out = draw_boxes(image, (5, 5, 20, 20), detection, ["bird"])
    assert out.shape == (60, 60, 3)


# ---------------------------------------------------------------------------
# save_thumbnail
# ---------------------------------------------------------------------------


def test_save_thumbnail_writes_200x200_jpeg(tmp_path):
    """save_thumbnail writes a 200x200 JPEG to disk."""
    roi = np.full((50, 40, 3), 120, dtype=np.uint8)
    out_path = tmp_path / "thumb.jpg"
    save_thumbnail(roi, str(out_path))
    assert out_path.exists()
    with Image.open(out_path) as img:
        assert img.size == (200, 200)
        assert img.format == "JPEG"
