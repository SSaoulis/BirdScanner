"""Tests for the tracking logger configuration and event helpers."""

import logging
from types import SimpleNamespace

from birdscanner.detector.track_logging import TrackingLogger, configure_logging


def _track():
    """A minimal track stand-in exposing the attributes the loggers read."""
    return SimpleNamespace(
        track_id=7,
        species="Robin",
        box=(1, 2, 3, 4),
        stable_frames=5,
        frames_since_seen=2,
    )


def test_tracking_logger_logs_stable_track(caplog):
    """log_stable_track emits an INFO record to the 'tracking' logger."""
    caplog.set_level(logging.INFO, logger="tracking")
    TrackingLogger().log_stable_track(_track())
    messages = [r.getMessage() for r in caplog.records]
    assert any("Track became stable" in m and "track_id=7" in m for m in messages)


def test_tracking_logger_logs_deleted_track(caplog):
    """log_deleted_track emits an INFO record naming the missing-frame count."""
    caplog.set_level(logging.INFO, logger="tracking")
    TrackingLogger().log_deleted_track(_track())
    messages = [r.getMessage() for r in caplog.records]
    assert any("Track deleted" in m and "missing_frames=2" in m for m in messages)


def test_configure_logging_sets_level_and_handler():
    """configure_logging sets the level and attaches a stdout stream handler."""
    logger = logging.getLogger("tracking")
    existing = list(logger.handlers)
    try:
        configure_logging(debug=True)
        assert logger.level == logging.DEBUG
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

        configure_logging(debug=False)
        assert logger.level == logging.INFO
    finally:
        # Restore the logger to its pre-test state so we don't leak handlers.
        logger.handlers = existing
        logger.setLevel(logging.NOTSET)
