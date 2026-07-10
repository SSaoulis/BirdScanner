"""Tests for the classification-pipeline wiring (``birdscanner/detector/gating.py``).

Focused on :func:`build_manager` bounding the async classification queue: an
unbounded queue held a full-resolution frame per backlog item and grew without
limit when the CPU classifier fell behind, eventually stalling the camera. These
tests need no camera, model, or DB — the ``onnxruntime`` stub required to import
the pipeline lives in the top-level ``conftest.py``.
"""

from typing import cast

from birdscanner.detector.gating import (
    CLASSIFICATION_QUEUE_MAXSIZE,
    Gating,
    build_manager,
)
from birdscanner.ml.best_frame import BestFrameSelector
from birdscanner.ml.classification import Classifier
from birdscanner.ml.tracking import StableDetectionTracker
from birdscanner.db.writer import DetectionWriter


def _gating() -> Gating:
    """Build a minimal gating bundle (no video recorder) for wiring tests."""
    return Gating(
        tracker=StableDetectionTracker(),
        best_frame_selector=BestFrameSelector(),
        video_recorder=None,
    )


def test_build_manager_bounds_the_async_queue(monkeypatch):
    """The async queue must be capped at ``CLASSIFICATION_QUEUE_MAXSIZE``.

    The queue holds full-resolution frames, so an unbounded queue leaked memory
    until the camera pipeline stalled. ``build_manager`` must size it so
    ``ClassificationManager.process`` drops excess frames under load.
    """
    monkeypatch.setattr("birdscanner.detector.gating.app_config.multithread", True)

    manager = build_manager(
        cast(Classifier, object()),
        _gating(),
        cast(DetectionWriter, object()),
    )
    try:
        assert manager.use_multithreading
        assert manager._queue is not None
        assert manager._queue.maxsize == CLASSIFICATION_QUEUE_MAXSIZE
    finally:
        manager.stop()
