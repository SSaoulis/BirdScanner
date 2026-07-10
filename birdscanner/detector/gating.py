"""Classification-pipeline wiring: stable-track gating and the manager.

Bridges the detector's runtime :mod:`config` to the platform-independent
classification pipeline under ``ml/``.  It builds the machinery that decides
*when* a track is classified and *how* the resulting work is dispatched:

* :func:`build_gating` — construct the stability tracker, the per-track
  best-frame selector, and (when video is enabled) the clip recorder, returning
  them in a :class:`Gating` bundle.
* :func:`build_manager` — assemble the :class:`ClassificationManager` and its
  :class:`PipelineContext` from a built :class:`Gating` bundle.

Keeping the picamera2/video wiring here (rather than in ``ml/``) preserves the
one-way ``detector -> ml`` layering: ``ml/`` receives the recorder callbacks as
plain injected callables.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from birdscanner.ml.tracking import (
    StableDetectionTracker,
    StableTrack,
)
from birdscanner.ml.best_frame import BestFrameSelector
from birdscanner.ml.classification_pipeline import (
    ClassificationManager,
    PipelineContext,
)
from birdscanner.ml.geomodel import GeoPriorAdjuster

from birdscanner.detector.config import config as app_config
from birdscanner.detector.track_logging import TrackingLogger
from birdscanner.detector.video_recorder import VideoRecorder
from birdscanner.db.database import SessionFactory
from birdscanner.db.geo_prior_store import load_geo_priors
from birdscanner.db.writer import DetectionWriter

logger = logging.getLogger("tracking")

# Upper bound on the async classification queue. The queue holds full-resolution
# frames (~5 MB each at ``DEFAULT_LONG_SIDE=1280``), so it must be bounded: when
# the CPU classifier falls behind a busy feeder, ``ClassificationManager.process``
# drops excess frames instead of growing without limit (an unbounded queue leaked
# memory until the camera pipeline stalled). Dropping is safe — the
# ``BestFrameSelector`` retains each track's best frame independently, so a stable
# track still classifies from its best observed frame even if intermediate
# detections are dropped. 32 caps in-flight RAM at ~160 MB worst case.
CLASSIFICATION_QUEUE_MAXSIZE = 32


@dataclass
class Gating:
    """Result of building the stable-track gating machinery.

    Attributes:
        tracker: The stability tracker.
        best_frame_selector: Per-track best-frame store.
        video_recorder: The clip recorder, or ``None`` when video is disabled.
    """

    tracker: StableDetectionTracker
    best_frame_selector: BestFrameSelector
    video_recorder: Optional[VideoRecorder]


def min_stable_frames(fps: int) -> int:
    """Frames a track must be stable before classification, from duration × fps."""
    return max(1, int((app_config.object_duration_threshold * fps) + 0.9999))


def build_gating(intrinsics: Any) -> Gating:
    """Build the stable-track gating machinery from the config and intrinsics.

    Constructs the tracker (wired to log lifecycle events and free per-track best
    frames), the best-frame selector, and — when video is enabled — the clip
    recorder.

    Args:
        intrinsics: The prepared network intrinsics (for the inference rate).

    Returns:
        The populated :class:`Gating` bundle.
    """
    fps = (
        app_config.intrinsics.fps
        or int(getattr(intrinsics, "inference_rate", 0) or 0)
        or 1
    )
    track_logger = TrackingLogger()
    selector = BestFrameSelector()

    def on_track_deleted(track: StableTrack) -> None:
        """Log the deletion and release the track's retained best frame."""
        track_logger.log_deleted_track(track)
        selector.discard(track.track_id)

    tracker = StableDetectionTracker(
        iou_threshold=0.6,
        min_stable_frames=min_stable_frames(fps),
        max_missing_frames=max(1, int(1.0 * fps)),
        on_track_became_stable=track_logger.log_stable_track,
        on_track_deleted=on_track_deleted,
    )

    recorder = (
        VideoRecorder(
            fps=fps,
            pre_roll_seconds=app_config.video.pre_roll_seconds,
            post_roll_seconds=app_config.video.post_roll_seconds,
        )
        if app_config.video.save
        else None
    )
    return Gating(tracker, selector, recorder)


def build_geo_adjuster(
    classifier: Any, session_factory: SessionFactory
) -> Optional[GeoPriorAdjuster]:
    """Build the runtime geomodel adjuster from the stored prior, if available.

    Reads the priors persisted at startup (see
    :mod:`birdscanner.detector.geo_priors`) and, together with the classifier's
    ``idx_to_class`` map, precomputes the reweighting matrix. Returns ``None`` —
    disabling the geomodel correction so classification is unchanged — when the
    classifier has no class-index map or no priors are stored (no location
    configured, or the startup rebuild was skipped/failed).

    Args:
        classifier: The species classifier (for its ``idx_to_class`` output map).
        session_factory: Zero-argument callable returning a ``Session`` context
            manager (to read the stored priors).

    Returns:
        A ready :class:`GeoPriorAdjuster`, or ``None`` when unavailable.
    """
    idx_to_class = getattr(classifier, "idx_to_class", None)
    if idx_to_class is None:
        return None
    priors = load_geo_priors(session_factory)
    if not priors:
        logger.info("Geomodel correction disabled: no stored geo priors.")
        return None
    logger.info("Geomodel correction enabled for %d species.", len(priors))
    return GeoPriorAdjuster(priors, idx_to_class)


def build_manager(
    classifier: Any,
    gating: Gating,
    detection_writer: DetectionWriter,
    geo_adjuster: Optional[GeoPriorAdjuster] = None,
) -> ClassificationManager:
    """Assemble the :class:`ClassificationManager` and its pipeline context.

    Args:
        classifier: The species classifier.
        gating: The stable-track gating bundle.
        detection_writer: The background DB writer.
        geo_adjuster: Optional geomodel Bayesian-update adjuster (see
            :func:`build_geo_adjuster`); ``None`` leaves classification unchanged.

    Returns:
        A ready manager with its results lock installed.
    """
    recorder = gating.video_recorder
    context = PipelineContext(
        classifier=classifier,
        tracker=gating.tracker,
        detection_writer=detection_writer,
        best_frame_selector=gating.best_frame_selector,
        record_fn=recorder.trigger if recorder is not None else None,
        video_frame_fn=recorder.add_frame if recorder is not None else None,
        geo_adjuster=geo_adjuster,
    )
    manager = ClassificationManager(
        context,
        use_multithreading=app_config.multithread,
        queue_maxsize=CLASSIFICATION_QUEUE_MAXSIZE,
    )
    return manager
