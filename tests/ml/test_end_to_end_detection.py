"""End-to-end detection pipeline test: from the classifier through to the DB.

The IMX500 on-chip object detector (the ``.rpk``, which only runs on the camera
silicon) is substituted by the hand-labelled bird boxes in
``tests/_test_images/bounding_box_locations.json`` — those boxes stand in for
what the ``.rpk`` would emit. From there the *real* pipeline runs end to end:

    Detection(box) -> ClassificationManager (stable-track gating, sync dispatch)
        -> preprocess_roi (crop the box to a padded square)
        -> Classifier.classify (real int8 ONNX)
        -> _persist_detection: save the still + thumbnail to IMAGE_DIR and write
           a DetectionRecord through the real DetectionWriter into SQLite.

This exercises everything downstream of object detection that runs off the Pi,
including the real DB round trip (background writer -> in-memory SQLite).

The ``real_classifier`` / ``bird_image_cases`` fixtures (``tests/ml/conftest.py``)
skip the module when the ONNX model or the JPEG fixtures are absent; the DB and
``image_dir`` fixtures come from the top-level ``tests/conftest.py``.
"""

import threading
from typing import List

import pytest
from sqlmodel import col, select

import birdscanner.ml.classification_pipeline as cp
from birdscanner.db.models import DetectionRecord
from birdscanner.db.writer import DetectionWriter
from birdscanner.ml.classification_pipeline import (
    NO_VIDEO_DISABLED,
    ClassificationManager,
    PipelineContext,
)
from birdscanner.ml.tracking import StableDetectionTracker

from .conftest import FakeDetection, ImageCase

# Stand-in YOLO/object-detection confidence for the substituted ``.rpk`` box; it
# is persisted verbatim as ``detection_confidence`` so we can assert on it.
_DETECTION_CONF = 0.87

# The pipeline only saves classifications above this confidence (the default
# ``PipelineContext.save_confidence_threshold`` /
# ``classification_pipeline.DEFAULT_SAVE_CONFIDENCE_THRESHOLD``); the fixture
# birds classify at ~0.9, well clear of it.
_SAVE_THRESHOLD = 0.4


def _run_detection(image, box, classifier, writer) -> StableDetectionTracker:
    """Drive one detection through the real stable-track gating pipeline.

    Builds a fresh single-frame-stable tracker for the detection, wires the real
    classifier + writer into a :class:`PipelineContext`, and dispatches the item
    synchronously through the production ``ClassificationManager`` path (gating
    on). ``IMAGE_DIR`` must already be patched by the caller.

    Args:
        image: Full-resolution RGB frame to classify.
        box: The bird box as ``(x, y, w, h)`` in frame pixels.
        classifier: The real ONNX classifier.
        writer: The real ``DetectionWriter`` persisting the result.

    Returns:
        The tracker, so callers can assert on the track's classified state.
    """
    detection = FakeDetection(box, conf=_DETECTION_CONF)
    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=1)
    tracker.update_frame([detection])  # detection_id 0 is now a stable track

    context = PipelineContext(
        classifier=classifier, tracker=tracker, detection_writer=writer
    )
    manager = ClassificationManager(context, use_stable_track_gating=True)
    manager.set_results_lock(threading.Lock())
    # The object-detection class is "bird" (what the .rpk reports) so species
    # classification is gated in for the stable track.
    manager.process((image, 0, detection, ["bird"], "bird"))
    return tracker


def test_pipeline_persists_classified_detections_to_db(
    real_classifier,
    bird_image_cases: List[ImageCase],
    session_factory,
    image_dir,
    monkeypatch,
):
    """Each labelled image is classified and persisted (row + files) end to end."""
    monkeypatch.setattr(cp, "IMAGE_DIR", str(image_dir))

    writer = DetectionWriter(session_factory)
    try:
        for case in bird_image_cases:
            _run_detection(case.image, case.box, real_classifier, writer)
    finally:
        writer.stop()  # flush the background queue before we read the DB

    with session_factory() as session:
        rows = session.exec(
            select(DetectionRecord).order_by(col(DetectionRecord.id))
        ).all()

    assert len(rows) == len(bird_image_cases)
    rows_by_species = {row.species: row for row in rows}

    for case in bird_image_cases:
        assert case.species in rows_by_species, (
            f"{case.name}: no persisted row for {case.species!r} "
            f"(got {sorted(rows_by_species)})"
        )
        row = rows_by_species[case.species]

        # Classification confidence cleared the save threshold and was persisted.
        assert row.confidence > _SAVE_THRESHOLD
        # The stand-in object-detection score round-tripped verbatim.
        assert row.detection_confidence == pytest.approx(_DETECTION_CONF)

        # The still + thumbnail were written under IMAGE_DIR/<species>/.
        assert row.image_path.startswith(f"{case.species}/")
        assert (image_dir / row.image_path).exists()
        assert (image_dir / row.thumbnail_path).exists()

        # The normalized detection box was persisted within [0, 1].
        for frac in (row.box_x, row.box_y, row.box_w, row.box_h):
            assert frac is not None and 0.0 <= frac <= 1.0

        # No clip was recorded (no recorder wired), and that reason was captured.
        assert row.video_path is None
        assert row.no_video_reason == NO_VIDEO_DISABLED


def test_zero_area_box_is_skipped_and_not_persisted(
    real_classifier,
    bird_image_cases: List[ImageCase],
    session_factory,
    image_dir,
    monkeypatch,
):
    """A degenerate zero-area box yields an empty ROI: skipped, not persisted.

    The track is left unclassified so a later, non-degenerate frame can still
    classify it — this is the exact guard that used to crash the classifier on an
    empty array.
    """
    monkeypatch.setattr(cp, "IMAGE_DIR", str(image_dir))
    case = bird_image_cases[0]
    tlx, tly, _, _ = case.box

    writer = DetectionWriter(session_factory)
    try:
        tracker = _run_detection(
            case.image, (tlx, tly, 0, 0), real_classifier, writer  # empty ROI
        )
    finally:
        writer.stop()

    with session_factory() as session:
        rows = session.exec(select(DetectionRecord)).all()
    assert rows == []

    track = tracker.track_for_detection_id(0)
    assert track is not None
    assert not track.classified
