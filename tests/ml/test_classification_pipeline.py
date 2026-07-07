"""Tests for the classification pipeline: dispatch robustness + the per-detection
processing core.

The ``onnxruntime`` stub needed to import this module lives in the top-level
``conftest.py``. The dispatch tests guard the regression where a detection that
raised inside the classifier killed the background worker permanently. The core
tests drive ``process_single_detection_with_stable_tracks`` / ``process_detections``
with the injected fakes from ``tests/ml/conftest.py`` — no camera, model, or DB.
"""

import json
import sys
import threading
import types
from typing import cast

import numpy as np
import pytest

import birdscanner.ml.classification_pipeline as cp
from birdscanner.ml.best_frame import BestFrameSelector
from birdscanner.ml.classification import Classifier
from birdscanner.ml.classification_pipeline import (
    ClassificationManager,
    PipelineContext,
)
from birdscanner.ml.geomodel import NUM_WEEKS, GeoPriorAdjuster
from birdscanner.ml.tracking import should_run_bird_classification_for_detection


def _make_item() -> tuple:
    """Build a minimal queue item; the dispatch target is patched, so contents don't matter."""
    return ("image", 0, object(), ["bird"], "bird")


# ---------------------------------------------------------------------------
# Dispatch robustness
# ---------------------------------------------------------------------------


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

    monkeypatch.setattr(
        cp, "process_single_detection_with_stable_tracks", fake_dispatch
    )

    manager = ClassificationManager(
        PipelineContext(classifier=cast(Classifier, object())),
        use_multithreading=True,
    )

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

    monkeypatch.setattr(
        cp, "process_single_detection_with_stable_tracks", fake_dispatch
    )

    manager = ClassificationManager(
        PipelineContext(classifier=cast(Classifier, object())),
        use_multithreading=False,
    )

    # Must not raise.
    manager.process(_make_item())


# ---------------------------------------------------------------------------
# process_single_detection_with_stable_tracks
# ---------------------------------------------------------------------------


def test_stable_track_classifies_saves_and_writes(
    tmp_path,
    monkeypatch,
    fake_detection,
    recording_writer,
    stable_tracker,
    frame_factory,
):
    """A stable bird detection classifies, saves still+thumbnail, and writes a row."""
    monkeypatch.setattr(cp, "IMAGE_DIR", str(tmp_path))
    det = fake_detection(box=(2, 2, 8, 8), conf=0.9, category=0)
    tracker = stable_tracker(det)
    frame = frame_factory(120, (32, 32))

    cp.process_single_detection_with_stable_tracks(
        (frame, 0, det, ["bird"], "bird"),
        PipelineContext(
            classifier=cast(Classifier, object()),
            tracker=tracker,
            classify_fn=lambda _classifier, _roi: ("Robin", 0.95),
            detection_writer=recording_writer,
        ),
    )

    assert len(recording_writer.writes) == 1
    write = recording_writer.writes[0]
    assert write.species == "Robin"
    assert write.confidence == 0.95
    assert write.detection_confidence == pytest.approx(0.9)  # from det.conf
    assert write.video_path is None  # no record_fn supplied
    assert write.no_video_reason == cp.NO_VIDEO_DISABLED  # no recorder wired
    # No geo_adjuster wired, so the geomodel debug fields stay unset.
    assert write.classifier_species is None
    assert write.classifier_confidence is None
    assert write.geo_scores is None

    species_dir = tmp_path / "Robin"
    assert len(list(species_dir.glob("*.png"))) == 1
    assert len(list(species_dir.glob("*_thumb.jpg"))) == 1


def test_stable_track_skips_degenerate_zero_area_box(
    tmp_path,
    monkeypatch,
    fake_detection,
    recording_writer,
    stable_tracker,
    frame_factory,
):
    """A zero-area ROI is skipped without classifying, saving, or marking the track."""
    monkeypatch.setattr(cp, "IMAGE_DIR", str(tmp_path))
    det = fake_detection(box=(2, 2, 0, 0), conf=0.9, category=0)
    tracker = stable_tracker(det)
    frame = frame_factory(120, (32, 32))
    classify_calls: list[int] = []

    def _classify(_c, _r):
        classify_calls.append(1)
        return ("Robin", 0.95)

    cp.process_single_detection_with_stable_tracks(
        (frame, 0, det, ["bird"], "bird"),
        PipelineContext(
            classifier=cast(Classifier, object()),
            tracker=tracker,
            classify_fn=_classify,
            detection_writer=recording_writer,
        ),
    )

    assert classify_calls == []  # classifier never invoked on an empty ROI
    assert recording_writer.writes == []
    # Track was left unclassified so a later, valid frame can still classify it.
    assert should_run_bird_classification_for_detection(0, tracker=tracker) is True


@pytest.mark.parametrize(
    "started,expect_video,expect_reason",
    [(True, True, None), (False, False, cp.NO_VIDEO_RECORDER_BUSY)],
)
def test_stable_track_video_path_only_when_recording_started(
    tmp_path,
    monkeypatch,
    fake_detection,
    recording_writer,
    recording_recorder,
    stable_tracker,
    frame_factory,
    started,
    expect_video,
    expect_reason,
):
    """video_path is persisted only when record_fn reports recording began."""
    monkeypatch.setattr(cp, "IMAGE_DIR", str(tmp_path))
    det = fake_detection(box=(2, 2, 8, 8), conf=0.9, category=0)
    tracker = stable_tracker(det)
    recorder = recording_recorder(started=started)

    cp.process_single_detection_with_stable_tracks(
        (frame_factory(120, (32, 32)), 0, det, ["bird"], "bird"),
        PipelineContext(
            classifier=cast(Classifier, object()),
            tracker=tracker,
            classify_fn=lambda _c, _r: ("Robin", 0.95),
            detection_writer=recording_writer,
            record_fn=recorder,
        ),
    )

    assert len(recorder.paths) == 1  # record_fn always attempted
    write = recording_writer.writes[0]
    assert (write.video_path is not None) is expect_video
    assert write.no_video_reason == expect_reason


def test_stable_track_uses_best_frame(
    tmp_path,
    monkeypatch,
    fake_detection,
    recording_writer,
    stable_tracker,
    frame_factory,
):
    """When a best frame is stored, it drives the ROI + persisted normalized box."""
    monkeypatch.setattr(cp, "IMAGE_DIR", str(tmp_path))
    det = fake_detection(box=(2, 2, 8, 8), conf=0.9, category=0)
    tracker = stable_tracker(det)
    track = tracker.track_for_detection_id(0)

    selector = BestFrameSelector()
    selector.observe(track.track_id, frame_factory(200, (32, 32)), (1, 1, 10, 10), 0.99)

    cp.process_single_detection_with_stable_tracks(
        (frame_factory(120, (32, 32)), 0, det, ["bird"], "bird"),
        PipelineContext(
            classifier=cast(Classifier, object()),
            tracker=tracker,
            classify_fn=lambda _c, _r: ("Robin", 0.95),
            detection_writer=recording_writer,
            best_frame_selector=selector,
        ),
    )

    # The selector was drained (its frame was taken for classification).
    assert selector.take(track.track_id) is None
    # The persisted normalized box comes from the best frame's box (1,1,10,10)/32.
    write = recording_writer.writes[0]
    assert write.box_x == pytest.approx(1 / 32)
    assert write.box_w == pytest.approx(10 / 32)


def test_stable_track_applies_geomodel_adjustment(
    tmp_path,
    monkeypatch,
    fake_detection,
    recording_writer,
    stable_tracker,
    frame_factory,
):
    """With a geo_adjuster, the posterior is persisted as species and both picks are stored."""
    monkeypatch.setattr(cp, "IMAGE_DIR", str(tmp_path))
    det = fake_detection(box=(2, 2, 8, 8), conf=0.9, category=0)
    tracker = stable_tracker(det)

    class _Classifier:
        """Fake classifier exposing the full-distribution API the geo path uses."""

        idx_to_class = {0: "Vagrant", 1: "Local robin"}

        def predict_proba(self, _roi):
            # Narrowly prefers the out-of-range Vagrant before the geomodel prior.
            return np.array([0.6, 0.4])

    # Vagrant is essentially absent locally; the Local robin is common — so the
    # posterior should flip to the robin.
    adjuster = GeoPriorAdjuster(
        {"Vagrant": [1e-3] * NUM_WEEKS, "Local robin": [0.9] * NUM_WEEKS},
        {0: "Vagrant", 1: "Local robin"},
        floor=1e-4,
        top_k=2,
    )

    cp.process_single_detection_with_stable_tracks(
        (frame_factory(120, (32, 32)), 0, det, ["bird"], "bird"),
        PipelineContext(
            classifier=cast(Classifier, _Classifier()),
            tracker=tracker,
            detection_writer=recording_writer,
            geo_adjuster=adjuster,
        ),
    )

    write = recording_writer.writes[0]
    assert write.species == "Local robin"  # geomodel-corrected posterior argmax
    assert write.confidence > 0.5
    assert write.classifier_species == "Vagrant"  # classifier's own pick
    assert write.classifier_confidence == pytest.approx(0.6)
    scores = json.loads(write.geo_scores)
    assert scores[0][0] == "Local robin"  # highest pre-normalised score
    assert scores[0][1] == pytest.approx(0.4 * 0.9)


def test_run_bird_classification_delegates_to_classifier():
    """run_bird_classification forwards to the classifier's classify()."""

    class _Classifier:
        def classify(self, _image):
            return ("Robin", 0.9)

    result = cp.run_bird_classification(
        cast(Classifier, _Classifier()), np.zeros((4, 4, 3), np.uint8)
    )
    assert result == ("Robin", 0.9)


# ---------------------------------------------------------------------------
# process_detections
# ---------------------------------------------------------------------------


def test_process_detections_feeds_video_observes_and_queues(
    monkeypatch, fake_detection, frame_factory, stable_tracker
):
    """process_detections feeds the pre-roll buffer, observes a best frame, and queues."""
    # Stub picamera2.MappedArray so the callback runs without the native binding.
    fake_picamera2 = types.ModuleType("picamera2")

    class _MappedArray:
        def __init__(self, request, _stream):
            self._m = request

        def __enter__(self):
            return self._m

        def __exit__(self, *exc):
            return False

    fake_picamera2.MappedArray = _MappedArray  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "picamera2", fake_picamera2)

    det = fake_detection(box=(2, 2, 8, 8), conf=0.9, category=0)
    tracker = stable_tracker(det)
    selector = BestFrameSelector()
    fed: list = []
    dispatched: list = []

    # Isolate process_detections: don't run the full classify path when queued.
    monkeypatch.setattr(
        cp,
        "process_single_detection_with_stable_tracks",
        lambda *a, **k: dispatched.append(1),
    )

    manager = ClassificationManager(
        PipelineContext(
            classifier=cast(Classifier, object()),
            tracker=tracker,
            best_frame_selector=selector,
            video_frame_fn=fed.append,
        ),
        use_multithreading=False,
    )

    frame = frame_factory(100, (32, 32))
    request = types.SimpleNamespace(array=frame)
    cp.process_detections(request, "main", [det], manager, ["bird"])

    assert len(fed) == 1  # clean frame fed to the video pre-roll buffer
    track = tracker.track_for_detection_id(0)
    assert selector.take(track.track_id) is not None  # best frame observed
    assert dispatched == [1]  # bird detection queued for classification


def test_process_detections_none_results_is_noop(fake_detection):
    """A ``None`` results list returns immediately (no frame access needed)."""
    # No picamera2 stub required: the function returns before importing it.
    cp.process_detections(
        object(), "main", None, cast(ClassificationManager, object()), ["bird"]
    )
