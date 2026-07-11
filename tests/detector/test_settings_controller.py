"""Tests for the runtime settings controller (live-apply + restart tracking)."""

import logging

import pytest

from birdscanner.ml import classification_pipeline
from birdscanner.ml.classification_pipeline import PipelineContext
from birdscanner.detector.config.config import config as app_config
from birdscanner.detector.config.settings import (
    default_settings,
    load_settings,
    merge_settings,
)
from birdscanner.detector.config.settings_controller import (
    SettingsController,
    apply_settings_to_config,
    apply_settings_to_context,
)


@pytest.fixture(autouse=True)
def _restore_globals():
    """Snapshot + restore the mutable global config/IMAGE_DIR around each test."""
    threshold = app_config.threshold
    excluded_classes = set(app_config.excluded_classes)
    duration = app_config.object_duration_threshold
    multithread = app_config.multithread
    debug = app_config.debug
    video_save = app_config.video.save
    pre_roll = app_config.video.pre_roll_seconds
    post_roll = app_config.video.post_roll_seconds
    image_dir = classification_pipeline.IMAGE_DIR
    level = logging.getLogger("tracking").level
    yield
    app_config.threshold = threshold
    app_config.excluded_classes = excluded_classes
    app_config.object_duration_threshold = duration
    app_config.multithread = multithread
    app_config.debug = debug
    app_config.video.save = video_save
    app_config.video.pre_roll_seconds = pre_roll
    app_config.video.post_roll_seconds = post_roll
    classification_pipeline.IMAGE_DIR = image_dir
    logging.getLogger("tracking").setLevel(level)


def _context() -> PipelineContext:
    """A minimal context (classifier unused by the controller)."""
    return PipelineContext(classifier=None)  # type: ignore[arg-type]


def test_apply_settings_to_config_pushes_all_fields() -> None:
    settings = merge_settings(
        default_settings(),
        {
            "detection_threshold": 0.7,
            "stability_seconds": 1.5,
            "multithread": False,
            "video_save": False,
            "video_pre_roll_seconds": 2.0,
            "image_dir": "/tmp/pics",
        },
    )
    apply_settings_to_config(settings)
    assert app_config.threshold == 0.7
    assert app_config.object_duration_threshold == 1.5
    assert app_config.multithread is False
    assert app_config.video.save is False
    assert app_config.video.pre_roll_seconds == 2.0
    assert classification_pipeline.IMAGE_DIR == "/tmp/pics"


def test_apply_settings_to_config_lowercases_excluded_classes() -> None:
    settings = merge_settings(
        default_settings(), {"excluded_classes": ["Bench", "PERSON"]}
    )
    apply_settings_to_config(settings)
    assert app_config.excluded_classes == {"bench", "person"}


def test_update_excluded_classes_applies_live(tmp_path) -> None:
    controller = SettingsController(
        str(tmp_path / "s.json"), default_settings(), _context()
    )
    state = controller.update({"excluded_classes": ["Car", "Bench"]})
    assert app_config.excluded_classes == {"car", "bench"}
    # It is a live field, so no restart is required.
    assert state["needs_restart"] is False


def test_apply_settings_to_context_lowercases_ignore() -> None:
    context = _context()
    settings = merge_settings(
        default_settings(), {"ignore_species": ["Robin", "UNKNOWN"]}
    )
    apply_settings_to_context(settings, context)
    assert context.ignore_species == {"robin", "unknown"}
    assert context.save_confidence_threshold == settings.classification_threshold


def test_get_state_reports_metadata(tmp_path) -> None:
    settings = default_settings()
    controller = SettingsController(str(tmp_path / "s.json"), settings, _context())
    state = controller.get_state()
    assert state["needs_restart"] is False
    assert "detection_threshold" in state["live_fields"]
    assert "stability_seconds" in state["restart_fields"]
    assert state["settings"]["image_dir"] == settings.image_dir


def test_update_live_field_applies_immediately(tmp_path) -> None:
    context = _context()
    path = str(tmp_path / "s.json")
    controller = SettingsController(path, default_settings(), context)

    state = controller.update({"detection_threshold": 0.9})

    assert app_config.threshold == 0.9  # applied live
    assert state["needs_restart"] is False  # live field: no restart
    assert load_settings(path).detection_threshold == 0.9  # persisted


def test_update_ignore_species_updates_context(tmp_path) -> None:
    context = _context()
    controller = SettingsController(
        str(tmp_path / "s.json"), default_settings(), context
    )
    controller.update({"ignore_species": ["Sparrow"]})
    assert context.ignore_species == {"sparrow"}


def test_update_restart_field_sets_needs_restart(tmp_path) -> None:
    controller = SettingsController(
        str(tmp_path / "s.json"), default_settings(), _context()
    )
    state = controller.update({"stability_seconds": 2.5})
    assert state["needs_restart"] is True


def test_update_debug_changes_log_level(tmp_path) -> None:
    controller = SettingsController(
        str(tmp_path / "s.json"), default_settings(), _context()
    )
    controller.update({"debug": True})
    assert logging.getLogger("tracking").level == logging.DEBUG
    controller.update({"debug": False})
    assert logging.getLogger("tracking").level == logging.INFO


def test_update_invalid_value_raises_and_does_not_persist(tmp_path) -> None:
    path = str(tmp_path / "s.json")
    controller = SettingsController(path, default_settings(), _context())
    with pytest.raises(ValueError):
        controller.update({"detection_threshold": 9.0})
    # State unchanged and nothing written.
    assert controller.get_state()["settings"]["detection_threshold"] == (
        default_settings().detection_threshold
    )
