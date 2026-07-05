"""The ``tracking`` logger: stream configuration and track-event helpers."""

import logging
import sys


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

    def __init__(self):
        """Create a logger bound to the ``tracking`` logger name."""
        self.logger = logging.getLogger("tracking")

    def log_stable_track(self, track):
        """Log that ``track`` has become stable."""
        self.logger.info(
            "Track became stable: track_id=%s species=%s box=%s stable_frames=%s",
            track.track_id,
            track.species,
            track.box,
            track.stable_frames,
        )

    def log_deleted_track(self, track):
        """Log that ``track`` has been deleted from the tracker."""
        self.logger.info(
            "Track deleted: track_id=%s species=%s box=%s stable_frames=%s missing_frames=%s",
            track.track_id,
            track.species,
            track.box,
            track.stable_frames,
            track.frames_since_seen,
        )
