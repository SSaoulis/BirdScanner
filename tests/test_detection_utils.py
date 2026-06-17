"""Unit tests for stateless helpers in ``src.detection_utils``."""

from src.detection_utils import label_for_category


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
