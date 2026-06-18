"""Unit tests for stateless helpers in ``src.detection_utils``."""

import pytest

from src.detection_utils import label_for_category, normalized_box


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
