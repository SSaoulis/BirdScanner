"""Multi-frame stability tracking for detections.

An IoU-based tracker that matches detections across consecutive frames and
gates classification until a track has been stable for ``min_stable_frames``
frames. Each track is classified at most once (see ``mark_classified``).
"""

from dataclasses import dataclass
from typing import Callable, Optional

from detection_utils import iou


@dataclass
class StableTrack:
    """A lightweight track for a single object across frames."""

    track_id: int
    box: tuple
    stable_frames: int = 1
    classified: bool = False
    frames_since_seen: int = 0
    species: Optional[str] = None


def match_detection_to_track(
    detection_box: tuple,
    tracks: list[StableTrack],
    *,
    iou_threshold: float,
) -> Optional[StableTrack]:
    """Return the best matching track for a detection, or None if no match."""

    best_track: Optional[StableTrack] = None
    best_iou = 0.0
    for t in tracks:
        score = iou(detection_box, t.box)
        if score > best_iou:
            best_iou = score
            best_track = t
    if best_track is None or best_iou <= iou_threshold:
        return None
    return best_track


def update_tracks_for_frame(
    detection_boxes: list[tuple],
    tracks: list[StableTrack],
    *,
    iou_threshold: float,
    max_missing_frames: int = 0,
    on_track_deleted: Optional[Callable[[StableTrack], None]] = None,
) -> tuple[list[StableTrack], dict[int, StableTrack]]:
    """Update track state from a list of detection boxes for a single frame.

    Args:
        detection_boxes: List of detection boxes for the frame.
        tracks: Existing tracks to update.
        iou_threshold: IoU threshold to consider a match.
        max_missing_frames: If > 0, delete tracks that have not been matched for this many frames.
        on_track_deleted: Optional callback invoked for each deleted track.

    Returns:
        (updated_tracks, per_detection_track_map)
    """

    per_det_track: dict[int, StableTrack] = {}
    used_track_ids: set[int] = set()

    # Age all tracks; any track that is matched this frame will be reset to 0.
    for t in tracks:
        t.frames_since_seen += 1

    next_track_id = (max((t.track_id for t in tracks), default=-1) + 1) if tracks else 0

    for det_id, box in enumerate(detection_boxes):
        match = match_detection_to_track(box, tracks, iou_threshold=iou_threshold)
        if match is None or match.track_id in used_track_ids:
            new_track = StableTrack(
                track_id=next_track_id,
                box=box,
                stable_frames=1,
                classified=False,
                frames_since_seen=0,
                species=None,
            )
            next_track_id += 1
            tracks.append(new_track)
            per_det_track[det_id] = new_track
            used_track_ids.add(new_track.track_id)
        else:
            match.box = box
            match.stable_frames += 1
            match.frames_since_seen = 0
            per_det_track[det_id] = match
            used_track_ids.add(match.track_id)

    # Drop tracks that haven't been seen recently.
    if max_missing_frames and max_missing_frames > 0:
        kept: list[StableTrack] = []
        for t in tracks:
            if t.frames_since_seen <= max_missing_frames:
                kept.append(t)
            else:
                if on_track_deleted is not None:
                    on_track_deleted(t)
        tracks = kept

    return tracks, per_det_track


def should_classify_track(track: StableTrack, *, min_stable_frames: int) -> bool:
    """Whether a track has been stable long enough and not yet classified."""

    return (not track.classified) and track.stable_frames >= min_stable_frames


class StableDetectionTracker:
    """Tracks detections across frames and gates classification until stable."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.6,
        min_stable_frames: int = 3,
        max_missing_frames: int = 0,
        on_track_became_stable: Optional[Callable[[StableTrack], None]] = None,
        on_track_deleted: Optional[Callable[[StableTrack], None]] = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_stable_frames = min_stable_frames
        self.max_missing_frames = max_missing_frames
        self.on_track_became_stable = on_track_became_stable
        self.on_track_deleted = on_track_deleted
        self._tracks: list[StableTrack] = []
        self._last_per_det_track: dict[int, StableTrack] = {}

    def update_frame(
        self,
        detections: list,
        *,
        keep_detection: Optional[Callable[[object], bool]] = None,
    ) -> None:
        """Update tracker state with a list of Detection-like objects.

        Args:
            detections: List of Detection-like objects (must have .box).
            keep_detection: Optional predicate; only detections for which this returns True
                contribute to tracking.
        """

        filtered = (
            [d for d in detections if keep_detection(d)]
            if keep_detection is not None
            else detections
        )

        boxes = [d.box for d in filtered]

        # Snapshot which tracks were already stable so we can detect first-time stability.
        previously_stable_ids = {
            t.track_id
            for t in self._tracks
            if t.stable_frames >= self.min_stable_frames
        }

        self._tracks, self._last_per_det_track = update_tracks_for_frame(
            boxes,
            self._tracks,
            iou_threshold=self.iou_threshold,
            max_missing_frames=self.max_missing_frames,
            on_track_deleted=self.on_track_deleted,
        )

        # Fire 'became stable' event once per track, at the moment it crosses the threshold.
        if self.on_track_became_stable is not None:
            for t in self._tracks:
                if (
                    t.track_id not in previously_stable_ids
                    and t.stable_frames >= self.min_stable_frames
                ):
                    self.on_track_became_stable(t)

    def track_for_detection_id(self, detection_id: int) -> Optional[StableTrack]:
        """Return the track last matched to ``detection_id``, or ``None``."""
        return self._last_per_det_track.get(detection_id)

    def mark_classified(self, track_id: int, *, species: Optional[str] = None) -> None:
        """Mark the track ``track_id`` as classified, optionally setting its species."""
        for t in self._tracks:
            if t.track_id == track_id:
                t.classified = True
                if species is not None:
                    t.species = species
                return

    def track_count(self) -> int:
        """Test helper: the number of currently active tracks."""

        return len(self._tracks)


# Global tracker instance used by default in the live pipeline.
# Tests can instantiate StableDetectionTracker directly.
stable_detection_tracker = StableDetectionTracker(
    iou_threshold=0.6, min_stable_frames=3
)


def should_run_bird_classification_for_detection(
    detection_id: int,
    *,
    tracker: StableDetectionTracker,
) -> bool:
    """New gating logic: only classify if the detection belongs to a stable track."""

    track = tracker.track_for_detection_id(detection_id)
    if track is None:
        return False
    return should_classify_track(track, min_stable_frames=tracker.min_stable_frames)
