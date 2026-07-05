"""Tests for the tracking-event logging helpers."""

import logging
from types import SimpleNamespace

from birdscanner.detector.track_logging import (
    TrackingLogger,
    on_track_became_stable,
    on_track_deleted,
)


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


def test_module_level_helpers_emit_debug(caplog):
    """The module-level helpers log at DEBUG without raising."""
    caplog.set_level(logging.DEBUG)
    on_track_became_stable(_track())
    on_track_deleted(_track())
    messages = [r.getMessage() for r in caplog.records]
    assert any("Track became stable" in m for m in messages)
    assert any("Track deleted" in m for m in messages)
