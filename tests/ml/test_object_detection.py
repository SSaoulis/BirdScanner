"""Unit tests for IMX500 inference-tensor parsing.

A fake IMX500 (with a stubbed ``convert_inference_coords``) and a fake intrinsics
object stand in for the camera hardware, so ``parse_detections`` can be exercised
against synthetic tensors.
"""

import numpy as np
import pytest

import birdscanner.ml.object_detection as od
from birdscanner.ml.object_detection import (
    Detection,
    InferenceContext,
    InferenceRoi,
    filter_included_detections,
    get_labels,
    parse_detections,
)


def _ctx(imx, intrinsics=None):
    """Bundle a fake IMX500 + intrinsics (+ a dummy picam2) for parse_detections."""
    return InferenceContext(imx, intrinsics or _Intrinsics(), object())


class _FakeIMX500:
    """Minimal IMX500 stand-in for parse_detections."""

    def __init__(self, outputs, input_size=(320, 320)):
        self._outputs = outputs
        self._input_size = input_size
        self.convert_calls = []

    def get_outputs(self, metadata, add_batch=True):
        return self._outputs

    def get_input_size(self):
        return self._input_size

    def convert_inference_coords(self, box, metadata, picam2):
        self.convert_calls.append(box)
        return (1, 2, 3, 4)


class _Intrinsics:
    """Minimal network-intrinsics stand-in."""

    def __init__(self, bbox_normalization=False, bbox_order="yx", labels=None):
        self.bbox_normalization = bbox_normalization
        self.bbox_order = bbox_order
        self.labels = labels or []


def _outputs(boxes, scores, classes):
    """Wrap arrays the way ``get_outputs(add_batch=True)`` returns them."""
    return [np.array([boxes]), np.array([scores]), np.array([classes])]


def test_parse_detections_filters_by_threshold_and_converts():
    """Only above-threshold detections survive and get converted coordinates."""
    boxes = np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]], dtype=np.float32)
    scores = np.array([0.9, 0.1], dtype=np.float32)
    classes = np.array([0, 1])
    imx = _FakeIMX500(_outputs(boxes, scores, classes))
    intrinsics = _Intrinsics(bbox_normalization=False, bbox_order="yx")

    result = parse_detections({}, _ctx(imx, intrinsics), threshold=0.5)

    assert len(result) == 1
    assert result[0].conf == pytest.approx(0.9)
    assert result[0].box == (1, 2, 3, 4)
    assert int(result[0].category) == 0


def test_parse_detections_none_outputs_returns_last_detections():
    """When the frame yields no inference output, the previous detections return."""
    od.last_detections = ["sentinel"]  # type: ignore[list-item]  # sentinel stands in for a Detection
    imx = _FakeIMX500(None)
    result = parse_detections({}, _ctx(imx), threshold=0.5)
    assert result == ["sentinel"]


def test_parse_detections_applies_normalization_and_xy_reorder():
    """bbox_normalization divides by input height and xy order swaps the axes."""
    boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)  # /100 -> 0.1,0.2,0.3,0.4
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([0])
    imx = _FakeIMX500(_outputs(boxes, scores, classes), input_size=(100, 100))
    intrinsics = _Intrinsics(bbox_normalization=True, bbox_order="xy")

    parse_detections({}, _ctx(imx, intrinsics), threshold=0.5)

    # Normalized to [0.1,0.2,0.3,0.4], then xy reorder [1,0,3,2] -> [0.2,0.1,0.4,0.3].
    passed_box = [float(v) for v in imx.convert_calls[0]]
    assert passed_box == pytest.approx([0.2, 0.1, 0.4, 0.3])


def test_parse_detections_remaps_roi_relative_box_to_full_sensor():
    """With an inference ROI, ROI-relative coords are remapped to full-sensor ones.

    When the on-chip DNN input is restricted to a crop, the network returns boxes
    normalized to the ROI. parse_detections must place them inside the ROI (and
    renormalize by the sensor size) *before* convert_inference_coords, or the box
    lands in the wrong place (the on-device regression this guards).
    """
    boxes = np.array([[0.5, 0.5, 0.6, 0.6]], dtype=np.float32)  # (y0,x0,y1,x1) in ROI
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([0])
    imx = _FakeIMX500(_outputs(boxes, scores, classes))
    intrinsics = _Intrinsics(bbox_normalization=False, bbox_order="yx")
    roi = InferenceRoi(
        left=1000, top=500, width=900, height=600, sensor_w=4000, sensor_h=3000
    )

    parse_detections({}, _ctx(imx, intrinsics), threshold=0.5, inference_roi=roi)

    # (roi_origin + frac * roi_size) / sensor_size, per axis.
    passed_box = [float(v) for v in imx.convert_calls[0]]
    assert passed_box == pytest.approx(
        [
            (500 + 0.5 * 600) / 3000,  # y0
            (1000 + 0.5 * 900) / 4000,  # x0
            (500 + 0.6 * 600) / 3000,  # y1
            (1000 + 0.6 * 900) / 4000,  # x1
        ]
    )
    # The remap actually moved the box (it is not the raw ROI-relative box).
    assert passed_box != pytest.approx([0.5, 0.5, 0.6, 0.6])


def test_parse_detections_without_roi_leaves_box_unchanged():
    """No inference ROI (full-FOV inference) passes the decoded box through as-is."""
    boxes = np.array([[0.5, 0.5, 0.6, 0.6]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    classes = np.array([0])
    imx = _FakeIMX500(_outputs(boxes, scores, classes))
    intrinsics = _Intrinsics(bbox_normalization=False, bbox_order="yx")

    parse_detections({}, _ctx(imx, intrinsics), threshold=0.5, inference_roi=None)

    passed_box = [float(v) for v in imx.convert_calls[0]]
    assert passed_box == pytest.approx([0.5, 0.5, 0.6, 0.6])


def test_get_labels_filters_empty_and_dash():
    """get_labels drops empty and placeholder '-' labels."""
    intrinsics = _Intrinsics(labels=["bird", "", "-", "cat"])
    assert get_labels(intrinsics) == ["bird", "cat"]


def test_detection_set_box():
    """set_box records the converted box; it starts as None."""
    detection = Detection(coords=np.zeros(4), category=1, conf=0.5, metadata={})
    assert detection.box is None
    detection.set_box((1, 2, 3, 4))
    assert detection.box == (1, 2, 3, 4)


def _det(category):
    """A Detection carrying only the class index (all filter needs)."""
    return Detection(coords=np.zeros(4), category=category, conf=0.9, metadata={})


_LABELS = ["bird", "bench", "person"]


def test_filter_included_keeps_only_matching_class_case_insensitively():
    """Only detections whose label is included (any case) survive."""
    dets = [_det(0), _det(1), _det(2)]  # bird, bench, person
    kept = filter_included_detections(dets, _LABELS, {"Bird"})
    assert [int(d.category) for d in kept] == [0]


def test_filter_included_empty_list_returns_input_unchanged():
    """An empty include set is a no-op that returns the same list object."""
    dets = [_det(0), _det(1)]
    assert filter_included_detections(dets, _LABELS, set()) is dets


def test_filter_included_keeps_out_of_range_category():
    """A category index outside the label list is kept (guarded downstream)."""
    dets = [_det(99)]
    assert filter_included_detections(dets, _LABELS, {"bird"}) == dets
