"""The ``tracking`` logger: stream configuration and track-event helpers."""

import logging
import sys
from typing import Optional

from birdscanner.ml.detection_utils import label_for_category


def configure_logging(debug: bool) -> None:
    """Configure the ``tracking`` logger to stream to stdout.

    Args:
        debug: When ``True`` the logger is set to DEBUG (track lifecycle events),
            otherwise INFO.
    """
    logger = logging.getLogger("tracking")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


class TrackingLogger:
    """Logs stable-track and track-deletion events to the ``tracking`` logger."""

    def __init__(self, labels: Optional[list] = None):
        """Create a logger bound to the ``tracking`` logger name.

        Args:
            labels: The object-detection (COCO) label list, used to resolve a
                deleted track's category index to a human class name. ``None``
                leaves the class reported as ``unknown``/``id:<n>``.
        """
        self.logger = logging.getLogger("tracking")
        self.labels = labels

    def _class_name(self, track) -> str:
        """Resolve a track's object-detection class to a log-friendly string.

        Returns the COCO label when it resolves, ``id:<n>`` when the track has a
        category index but no matching label, or ``unknown`` when no category was
        recorded on the track.
        """
        category = getattr(track, "category", None)
        if category is None:
            return "unknown"
        label = (
            label_for_category(self.labels, int(category))
            if self.labels is not None
            else None
        )
        return label if label is not None else f"id:{int(category)}"

    def log_stable_track(self, track):
        """Log that ``track`` has become stable."""
        self.logger.info(
            "Track became stable: track_id=%s stable_frames=%s",
            track.track_id,
            track.stable_frames,
        )

    def log_deleted_track(self, track):
        """Log that ``track`` has been deleted from the tracker."""
        self.logger.info(
            "Track deleted: track_id=%s stable_frames=%s missing_frames=%s class=%s",
            track.track_id,
            track.stable_frames,
            track.frames_since_seen,
            self._class_name(track),
        )
