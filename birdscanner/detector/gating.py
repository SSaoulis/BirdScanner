"""Classification-pipeline wiring: stable-track gating and the manager.

Bridges the detector's runtime :mod:`config` to the platform-independent
classification pipeline under ``ml/``.  It builds the machinery that decides
*when* a track is classified and *how* the resulting work is dispatched:

* :func:`build_gating` — construct the stability tracker, the per-track
  best-frame selector, and (when video is enabled) the clip recorder, returning
  them in a :class:`Gating` bundle.  With ``object_duration_threshold <= 0`` it
  returns the legacy per-frame configuration where every component is ``None``.
* :func:`build_manager` — assemble the :class:`ClassificationManager` and its
  :class:`PipelineContext` from a built :class:`Gating` bundle.

Keeping the picamera2/video wiring here (rather than in ``ml/``) preserves the
one-way ``detector -> ml`` layering: ``ml/`` receives the recorder callbacks as
plain injected callables.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

from birdscanner.ml.tracking import (
    StableDetectionTracker,
    StableTrack,
    stable_detection_tracker,
)
from birdscanner.ml.best_frame import BestFrameSelector
from birdscanner.ml.classification_pipeline import (
    ClassificationManager,
    PipelineContext,
)

from birdscanner.detector.config import config as app_config
from birdscanner.detector.track_logging import TrackingLogger
from birdscanner.detector.video_recorder import VideoRecorder
from birdscanner.db.writer import DetectionWriter

logger = logging.getLogger("tracking")


@dataclass
class Gating:
    """Result of building the stable-track gating machinery.

    Attributes:
        use_stable_tracks: Whether stable-track gating is active.
        tracker: The stability tracker; the module-global default in legacy
            per-frame mode (where it is unused).
        best_frame_selector: Per-track best-frame store, or ``None`` when gating
            is off.
        video_recorder: The clip recorder, or ``None`` when gating is off or
            video is disabled.
    """

    use_stable_tracks: bool
    tracker: StableDetectionTracker
    best_frame_selector: Optional[BestFrameSelector]
    video_recorder: Optional[VideoRecorder]


def min_stable_frames(fps: int) -> int:
    """Frames a track must be stable before classification, from duration × fps."""
    return max(1, int((app_config.object_duration_threshold * fps) + 0.9999))


def build_gating(intrinsics: Any) -> Gating:
    """Build the stable-track gating machinery from the config and intrinsics.

    When ``object_duration_threshold <= 0`` gating is disabled (legacy per-frame
    mode) and every returned component is ``None``. Otherwise this constructs the
    tracker (wired to log lifecycle events and free per-track best frames), the
    best-frame selector, and — when video is enabled — the clip recorder.

    Args:
        intrinsics: The prepared network intrinsics (for the inference rate).

    Returns:
        The populated :class:`Gating` bundle.
    """
    if not app_config.object_duration_threshold > 0:
        return Gating(False, stable_detection_tracker, None, None)

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
    return Gating(True, tracker, selector, recorder)


def build_manager(
    classifier: Any, gating: Gating, detection_writer: DetectionWriter
) -> ClassificationManager:
    """Assemble the :class:`ClassificationManager` and its pipeline context.

    Args:
        classifier: The species classifier.
        gating: The stable-track gating bundle.
        detection_writer: The background DB writer.

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
    )
    manager = ClassificationManager(
        context,
        use_multithreading=app_config.multithread,
        use_stable_track_gating=gating.use_stable_tracks,
    )
    manager.set_results_lock(threading.Lock())
    return manager
