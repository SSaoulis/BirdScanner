"""Unit tests for the fake ``picamera2`` / ``libcamera`` stand-ins.

These exercise the fakes in isolation (without the real camera module), covering
the IMX500 tensor convention, the name->label-index mapping, the ScalerCrop
coordinate conversion, the frame-pump lifecycle, and the writable MappedArray.
"""

import numpy as np
import pytest

from birdscanner.detector.emulation.fakes import (
    FakeMappedArray,
    FakePicam2,
    FakeRequest,
    build_emulation_state,
    set_emulation_state,
)
from birdscanner.detector.emulation.yolo import Detected


class _SolidSource:
    """Yields a fixed solid frame each call."""

    def __init__(self, width=640, height=480):
        self._frame = np.full((height, width, 3), 100, dtype=np.uint8)

    def next_frame(self):
        return self._frame.copy()


class _FixedDetector:
    """Emits a preset list of detections regardless of the frame."""

    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        del frame
        return list(self._detections)


@pytest.fixture()
def emulation_state():
    """Install an emulation state for the fakes and clear it afterwards."""
    detector = _FixedDetector([Detected((0.25, 0.5, 0.75, 1.0), 0.8, "bird")])
    state = build_emulation_state(_SolidSource(), detector, max_frames=3)
    # Give the shared IMX500 a label list (prepare_intrinsics would do this).
    state.imx500.network_intrinsics.labels = ["person", "bird", "cat"]
    yield state
    set_emulation_state(None)


def test_get_outputs_uses_rpk_tensor_convention(emulation_state):
    """get_outputs emits [boxes(1,N,4), scores, classes] scaled to input_h, xy order."""
    imx500 = emulation_state.imx500
    imx500.current_detections = [Detected((0.25, 0.5, 0.75, 1.0), 0.8, "bird")]

    boxes, scores, classes = imx500.get_outputs({}, add_batch=True)

    assert boxes.shape == (1, 1, 4)
    assert scores.shape == (1, 1)
    input_h = imx500.get_input_size()[1]
    # xyxy scaled by input_h, in xy order (x0, y0, x1, y1).
    np.testing.assert_allclose(
        boxes[0, 0], [0.25 * input_h, 0.5 * input_h, 0.75 * input_h, 1.0 * input_h]
    )
    assert scores[0, 0] == pytest.approx(0.8)
    # "bird" maps to index 1 in the intrinsics labels, not its COCO index.
    assert int(classes[0, 0]) == 1


def test_get_outputs_drops_unknown_labels(emulation_state):
    """A detection whose class name is not in the label list is dropped."""
    imx500 = emulation_state.imx500
    imx500.current_detections = [Detected((0.1, 0.1, 0.2, 0.2), 0.9, "airplane")]

    boxes, scores, classes = imx500.get_outputs({})

    assert boxes.shape == (1, 0, 4)
    assert scores.shape == (1, 0)
    assert classes.shape == (1, 0)


def test_convert_inference_coords_maps_to_main_pixels(emulation_state):
    """A normalized (y0, x0, y1, x1) box scales to main-stream (x, y, w, h) pixels."""
    imx500 = emulation_state.imx500
    picam2 = FakePicam2()
    picam2.main_size = (1000, 500)

    # box in decoded (y0, x0, y1, x1) order: top=0.2, left=0.1, bottom=0.6, right=0.4
    box = (0.2, 0.1, 0.6, 0.4)
    result = imx500.convert_inference_coords(box, {}, picam2)

    assert result == (
        100,
        100,
        300,
        200,
    )  # x=0.1*1000, y=0.2*500, w=0.3*1000, h=0.4*500


def test_capture_metadata_fires_callback_and_counts(emulation_state):
    """capture_metadata pulls a frame, runs the detector, and fires pre_callback."""
    picam2 = FakePicam2()
    picam2.main_size = (320, 240)
    received: list = []
    picam2.pre_callback = received.append

    picam2.capture_metadata()

    assert len(received) == 1
    request = received[0]
    frame = request.make_array("main")
    assert frame.shape == (240, 320, 3)  # resized to main_size (h, w, 3)
    # The detector ran and stashed the detection on the shared IMX500.
    assert emulation_state.imx500.current_detections


def test_capture_metadata_raises_at_max_frames(emulation_state):
    """The frame budget (max_frames=3) unwinds the loop via KeyboardInterrupt."""
    picam2 = FakePicam2()
    picam2.main_size = (64, 64)

    picam2.capture_metadata()
    picam2.capture_metadata()
    with pytest.raises(KeyboardInterrupt):
        picam2.capture_metadata()


def test_mapped_array_exposes_writable_frame():
    """FakeMappedArray yields the request's frame and accepts writes."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    request = FakeRequest(frame, scaler=(0, 0, 4, 4))

    with FakeMappedArray(request, "main") as mapped:
        assert mapped.array is frame
        mapped.array[:] = 5

    assert np.all(frame == 5)


def test_fake_imx500_requires_installed_state():
    """Constructing a FakePicam2 without an installed state raises."""
    set_emulation_state(None)
    with pytest.raises(RuntimeError):
        FakePicam2()
