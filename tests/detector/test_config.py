"""Tests for the application configuration dataclasses."""

import pytest

from birdscanner.detector.config import Config, IntrinsicsConfig, VideoConfig, config


def test_config_defaults_mirror_known_good_runtime():
    """The defaults match the documented known-good YOLO11n invocation."""
    c = Config()
    assert c.model.endswith(".rpk")
    assert c.intrinsics.bbox_order == "xy"
    assert c.intrinsics.bbox_normalization is True
    assert c.threshold == pytest.approx(0.55)
    assert c.multithread is True
    assert c.object_duration_threshold == pytest.approx(0.1)
    assert c.video.save is True
    assert c.video.pre_roll_seconds == pytest.approx(3.0)
    assert c.video.post_roll_seconds == pytest.approx(4.0)


def test_module_config_is_a_config_instance():
    """The module-level singleton is a ready-to-use Config."""
    assert isinstance(config, Config)


def test_sub_configs_are_independent_per_instance():
    """Each Config gets its own sub-config instances (no shared mutable default)."""
    a, b = Config(), Config()
    assert a.intrinsics is not b.intrinsics
    assert a.video is not b.video
    assert isinstance(a.intrinsics, IntrinsicsConfig)
    assert isinstance(a.video, VideoConfig)


def test_optional_intrinsic_fields_default_to_none():
    """Optional intrinsic overrides default to None.

    ``main.py``'s override loop iterates ``vars(config.intrinsics)`` and skips
    ``None`` values so they never clobber the network-intrinsics defaults.
    """
    intr = IntrinsicsConfig()
    for name in ("fps", "ignore_dash_labels", "preserve_aspect_ratio", "labels"):
        assert getattr(intr, name) is None


def test_intrinsics_vars_exposes_override_fields():
    """vars(config.intrinsics) exposes every field the override loop iterates."""
    keys = set(vars(IntrinsicsConfig()))
    assert {
        "fps",
        "bbox_normalization",
        "bbox_order",
        "ignore_dash_labels",
        "preserve_aspect_ratio",
        "labels",
    } == keys
