"""Robustness tests for ``ClassificationManager`` dispatch.

These guard the regression where a single detection that raised inside the
classifier (e.g. a degenerate, zero-area ROI) propagated out of the background
worker thread and killed it permanently — silently stopping all further
classification and DB writes.
"""

import sys
import threading
import types

# ``classification_pipeline`` transitively imports ``onnxruntime`` (a Pi/CI-only
# dependency) at module load. These tests patch out the classifier dispatch, so a
# stub module is enough to import the pipeline anywhere; ``setdefault`` leaves a
# real install untouched on the Pi/CI.
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

import src.classification_pipeline as cp  # noqa: E402
from src.classification_pipeline import ClassificationManager  # noqa: E402


def _make_item() -> tuple:
    """Build a minimal queue item; the dispatch target is patched, so contents don't matter."""

    return ("image", 0, object(), ["bird"], "bird")


def test_async_worker_survives_a_raising_detection(monkeypatch):
    """A detection that raises must be logged and skipped, not kill the worker."""

    calls: list[int] = []
    done = threading.Event()

    def fake_dispatch(item, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("simulated classifier failure on a bad ROI")
        if len(calls) == 2:
            done.set()

    monkeypatch.setattr(cp, "process_single_detection_with_stable_tracks", fake_dispatch)

    manager = ClassificationManager(
        classifier=object(),
        use_multithreading=True,
        use_stable_track_gating=True,
    )
    manager.set_results_lock(threading.Lock())

    try:
        manager.process(_make_item())  # raises inside the worker
        manager.process(_make_item())  # must still be processed
        assert done.wait(timeout=5.0), "worker thread died after the first exception"
        assert len(calls) == 2
    finally:
        manager.stop()


def test_sync_dispatch_swallows_exceptions(monkeypatch):
    """In sync mode a raising detection must not propagate into the camera callback."""

    def fake_dispatch(item, **_kwargs):
        raise ValueError("simulated classifier failure")

    monkeypatch.setattr(cp, "process_single_detection_with_stable_tracks", fake_dispatch)

    manager = ClassificationManager(
        classifier=object(),
        use_multithreading=False,
        use_stable_track_gating=True,
    )
    manager.set_results_lock(threading.Lock())

    # Must not raise.
    manager.process(_make_item())
