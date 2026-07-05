"""Fixtures for the ML pipeline tests.

The classification pipeline is built around dependency injection (classifier,
writer, best-frame selector, record/video callables are all passed in), so these
lightweight fakes exercise the real orchestration without a camera, an ONNX model,
or a database.
"""

from typing import Any, List, Optional, Tuple

import pytest


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
