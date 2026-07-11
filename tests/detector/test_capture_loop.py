"""End-to-end capture-loop test driven by the emulated IMX500 camera.

This is the payoff of the emulation package: with the fake ``picamera2`` /
``libcamera`` modules installed, the **real** camera bring-up
(``wait_for_camera`` -> ``prepare_intrinsics`` -> ``build_camera``) and the
**real** ``_run_capture_loop`` execute off-Pi.  A deterministic
:class:`FixedBoxDetector` stands in for stage 1 (no ONNX model needed), so the
test is reproducible and model-independent.

It asserts the full seam works: frames -> fake IMX500 tensor -> real
``parse_detections`` -> real ``process_detections`` callback -> stable-track
gating -> classification (stubbed) -> a persisted ``DetectionRecord``.  This
covers ``detector/camera.py`` (previously 0%) and ``main.py``'s loop/callback.
"""

from typing import Any, List, cast

import numpy as np
import pytest

import birdscanner.ml.classification_pipeline as cp
from birdscanner.db.writer import DetectionWriter
from birdscanner.detector.config.config import config as app_config
from birdscanner.ml.classification import Classifier


class _RecordingWriter:
    """Captures written ``DetectionRecord``s instead of persisting them."""

    def __init__(self) -> None:
        """Start with an empty capture log."""
        self.writes: List[Any] = []

    def write(self, record: Any) -> None:
        """Record one written ``DetectionRecord``."""
        self.writes.append(record)


def test_capture_loop_persists_emulated_bird(camera_emulator, monkeypatch, tmp_path):
    """A bird detected every frame flows through the real loop to a written record."""
    from .conftest import FixedBoxDetector, SolidFrameSource

    # Deterministic, sync, no video clip — keep the test fast and model-free.
    monkeypatch.setattr(app_config, "multithread", False)
    monkeypatch.setattr(app_config.video, "save", False)
    monkeypatch.setattr(cp, "IMAGE_DIR", str(tmp_path))

    camera_emulator(SolidFrameSource(), FixedBoxDetector(), max_frames=8)

    # Import only after the fakes are installed.
    from birdscanner.detector.hardware.camera import (
        build_camera,
        prepare_intrinsics,
        wait_for_camera,
    )
    from birdscanner.detector.pipeline.gating import build_gating
    from birdscanner.detector.main import _run_capture_loop
    from birdscanner.ml.classification_pipeline import (
        ClassificationManager,
        PipelineContext,
    )

    imx500 = wait_for_camera("fake.rpk")
    intrinsics = prepare_intrinsics(imx500)
    gating = build_gating(intrinsics)
    camera = build_camera(imx500, intrinsics)

    writer = _RecordingWriter()
    context = PipelineContext(
        classifier=cast(Classifier, object()),
        tracker=gating.tracker,
        classify_fn=lambda _classifier, _roi: ("Robin", 0.9),
        detection_writer=cast(DetectionWriter, writer),
        best_frame_selector=gating.best_frame_selector,
    )
    manager = ClassificationManager(context, use_multithreading=False)

    # The real loop runs until the fake camera raises KeyboardInterrupt at
    # max_frames (the same signal main() shuts down on).
    with pytest.raises(KeyboardInterrupt):
        _run_capture_loop(camera, manager, gating)

    assert writer.writes, "no detection was classified + persisted through the loop"
    record = writer.writes[0]
    assert record.species == "Robin"
    # The stand-in YOLO confidence round-tripped as detection_confidence.
    assert record.detection_confidence == pytest.approx(0.95)
    # The normalized box was persisted within [0, 1].
    for frac in (record.box_x, record.box_y, record.box_w, record.box_h):
        assert frac is not None and 0.0 <= frac <= 1.0

    # The track was marked classified exactly once (gating held).
    track = gating.tracker.track_for_detection_id(0)
    assert track is not None and track.classified


def test_capture_loop_stops_when_frames_exhausted(camera_emulator, monkeypatch):
    """An exhausting frame source unwinds the loop via KeyboardInterrupt."""
    from .conftest import FixedBoxDetector

    monkeypatch.setattr(app_config, "multithread", False)
    monkeypatch.setattr(app_config.video, "save", False)

    class _OneShotSource:
        """Yields a single frame, then reports exhaustion."""

        def __init__(self) -> None:
            self._served = False

        def next_frame(self):
            if self._served:
                return None
            self._served = True
            return np.full((480, 640, 3), 90, dtype=np.uint8)

    camera_emulator(_OneShotSource(), FixedBoxDetector())

    from birdscanner.detector.hardware.camera import (
        build_camera,
        prepare_intrinsics,
        wait_for_camera,
    )
    from birdscanner.detector.pipeline.gating import build_gating
    from birdscanner.detector.main import _run_capture_loop
    from birdscanner.ml.classification_pipeline import (
        ClassificationManager,
        PipelineContext,
    )

    imx500 = wait_for_camera("fake.rpk")
    intrinsics = prepare_intrinsics(imx500)
    gating = build_gating(intrinsics)
    camera = build_camera(imx500, intrinsics)
    context = PipelineContext(
        classifier=cast(Classifier, object()),
        tracker=gating.tracker,
        classify_fn=lambda _classifier, _roi: ("Robin", 0.9),
        best_frame_selector=gating.best_frame_selector,
    )
    manager = ClassificationManager(context, use_multithreading=False)

    with pytest.raises(KeyboardInterrupt):
        _run_capture_loop(camera, manager, gating)
