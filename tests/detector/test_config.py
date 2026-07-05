"""Tests for the application configuration dataclass."""

import pytest

from birdscanner.detector.config import Config, config


def test_config_defaults_mirror_known_good_runtime():
    """The defaults match the documented known-good YOLO11n invocation."""
    c = Config()
    assert c.model.endswith(".rpk")
    assert c.bbox_order == "xy"
    assert c.bbox_normalization is True
    assert c.threshold == pytest.approx(0.55)
    assert c.multithread is True
    assert c.object_duration_threshold == pytest.approx(0.1)
    assert c.save_video is True
    assert c.video_pre_roll_seconds == pytest.approx(3.0)
    assert c.video_post_roll_seconds == pytest.approx(4.0)


def test_module_config_is_a_config_instance():
    """The module-level singleton is a ready-to-use Config."""
    assert isinstance(config, Config)


def test_optional_intrinsic_fields_default_to_none():
    """Optional intrinsic overrides default to None.

    ``main.py``'s override loop iterates ``vars(config)`` and skips ``None`` values
    so they never clobber the network-intrinsics defaults.
    """
    c = Config()
    for field in ("fps", "ignore_dash_labels", "preserve_aspect_ratio", "labels"):
        assert getattr(c, field) is None


def test_vars_exposes_all_runtime_settings():
    """vars(config) exposes every setting the override loop iterates."""
    keys = set(vars(Config()))
    assert {
        "model",
        "fps",
        "bbox_normalization",
        "bbox_order",
        "threshold",
        "multithread",
        "object_duration_threshold",
        "save_video",
        "video_pre_roll_seconds",
        "video_post_roll_seconds",
    } <= keys
