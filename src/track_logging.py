"""Logging helpers for stable-track and track-deletion events."""

import logging


def on_track_became_stable(track):
    """Log that ``track`` has become stable (species may still be unknown)."""
    # Species may be unknown until classification runs; log what we have.
    logging.debug(
        "Track became stable: track_id=%s species=%s box=%s stable_frames=%s",
        track.track_id,
        track.species,
        track.box,
        track.stable_frames,
    )


def on_track_deleted(track):
    """Log that ``track`` has been deleted from the tracker."""
    logging.debug(
        "Track deleted: track_id=%s species=%s box=%s stable_frames=%s missing_frames=%s",
        track.track_id,
        track.species,
        track.box,
        track.stable_frames,
        track.frames_since_seen,
    )


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
