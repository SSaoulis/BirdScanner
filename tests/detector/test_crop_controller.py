"""Tests for CropController driven by a fake picamera2.

The controller's geometry + persistence live in ``crop`` (unit-tested without a
camera); here a ``_FakePicam2`` records the camera calls so the live-apply vs.
reconfigure branching and the full-sensor preview capture can be exercised.
"""

import numpy as np
import pytest

from birdscanner.detector.crop import (
    SENSOR_H,
    SENSOR_W,
    default_crop_region,
    main_stream_size_for_crop,
)
from birdscanner.detector.crop_controller import CropController


class _FakeRequest:
    """Fake picamera2 capture request."""

    def __init__(self, frame, scaler):
        self._frame = frame
        self._scaler = scaler
        self.released = False

    def get_metadata(self):
        return {"ScalerCrop": self._scaler}

    def make_array(self, _stream):
        return self._frame

    def release(self):
        self.released = True


class _FakePicam2:
    """Fake Picamera2 recording every camera interaction the controller makes."""

    def __init__(self, frame, settle=True):
        self.frame = frame
        self.settle = settle
        self.controls: list = []
        self.configs: list = []
        self.stopped = 0
        self.started = 0
        self._last_scaler = None

    def set_controls(self, controls):
        self.controls.append(controls)
        if "ScalerCrop" in controls:
            self._last_scaler = controls["ScalerCrop"]

    def stop(self):
        self.stopped += 1

    def configure(self, config):
        self.configs.append(config)

    def start(self):
        self.started += 1

    def capture_request(self):
        # When settling, report the just-set crop so _capture_settled returns; when
        # not settling, report a crop that never matches the requested target.
        scaler = self._last_scaler if self.settle else (0, 0, 1, 1)
        return _FakeRequest(self.frame, scaler)


def _controller(tmp_path, cam, region=None):
    """Build a CropController around ``cam`` at the default (or given) region."""
    region = region or default_crop_region()
    main_size = main_stream_size_for_crop(region.w, region.h)
    return CropController(
        cam,
        region,
        main_size,
        config_factory=lambda main, scaler: ("config", main, scaler),
        config_path=str(tmp_path / "crop.json"),
        sensor_w=SENSOR_W,
        sensor_h=SENSOR_H,
    )


def test_get_state_returns_sensor_pixels_and_normalized(tmp_path, frame_factory):
    """get_state reports sensor px + a normalized box + the sensor dimensions."""
    region = default_crop_region()
    ctrl = _controller(tmp_path, _FakePicam2(frame_factory(0, (8, 8))), region)
    state = ctrl.get_state()
    assert (state["x"], state["y"], state["w"], state["h"]) == (
        region.x,
        region.y,
        region.w,
        region.h,
    )
    assert set(state["norm"]) == {"nx", "ny", "nw", "nh"}
    assert state["sensor_w"] == SENSOR_W and state["sensor_h"] == SENSOR_H


def test_reset_to_default_same_aspect_applies_live(tmp_path, frame_factory):
    """Resetting to the default (same aspect) applies live — no stop/reconfigure."""
    cam = _FakePicam2(frame_factory(0, (8, 8)))
    ctrl = _controller(tmp_path, cam)
    ctrl.reset_to_default()
    assert cam.stopped == 0 and cam.started == 0
    assert any("ScalerCrop" in c for c in cam.controls)


def test_aspect_change_triggers_reconfigure_and_persists(tmp_path, frame_factory):
    """An aspect-ratio change stops/reconfigures/starts the camera and persists JSON."""
    cam = _FakePicam2(frame_factory(0, (8, 8)))
    ctrl = _controller(tmp_path, cam)
    # A full-width, half-height box changes the aspect ratio from the square default.
    ctrl.set_from_normalized(0.0, 0.0, 1.0, 0.5)
    assert cam.stopped == 1 and cam.started == 1
    assert cam.configs  # config_factory output was applied
    assert (tmp_path / "crop.json").exists()


def test_capture_full_preview_restores_previous_crop(tmp_path, frame_factory):
    """The full-sensor preview widens the crop then restores the previous one."""
    region = default_crop_region()
    cam = _FakePicam2(frame_factory(7, (8, 8)))
    ctrl = _controller(tmp_path, cam, region)

    arr = ctrl.capture_full_preview_array()
    assert isinstance(arr, np.ndarray)

    full = (0, 0, SENSOR_W, SENSOR_H)
    previous = region.as_tuple()
    # It widened to full, then the final control restored the previous crop.
    assert {"ScalerCrop": full} in cam.controls
    assert cam.controls[-1] == {"ScalerCrop": previous}


def test_capture_settled_raises_when_no_frame(tmp_path):
    """_capture_settled raises if the camera never produces a settled frame."""
    cam = _FakePicam2(frame=None, settle=False)
    ctrl = _controller(tmp_path, cam)
    with pytest.raises(RuntimeError):
        ctrl.capture_full_preview_array()
