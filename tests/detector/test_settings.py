"""Tests for the pure settings domain (validation + JSON persistence)."""

import json
from dataclasses import fields

import pytest

from birdscanner.detector.settings import (
    LIVE_FIELDS,
    RESTART_FIELDS,
    Settings,
    default_settings,
    load_settings,
    merge_settings,
    save_settings,
    settings_config_path,
)


def test_default_settings_has_all_fields() -> None:
    settings = default_settings()
    assert isinstance(settings, Settings)
    assert isinstance(settings.ignore_species, list)
    assert 0.0 <= settings.detection_threshold <= 1.0
    assert 0.0 <= settings.classification_threshold <= 1.0
    assert settings.image_dir


def test_live_and_restart_fields_partition_the_model() -> None:
    all_fields = {f.name for f in fields(Settings)}
    assert LIVE_FIELDS | RESTART_FIELDS == all_fields
    assert not LIVE_FIELDS & RESTART_FIELDS


def test_merge_applies_valid_updates() -> None:
    base = default_settings()
    merged = merge_settings(base, {"detection_threshold": 0.7, "multithread": False})
    assert merged.detection_threshold == 0.7
    assert merged.multithread is False
    # Unchanged fields are preserved; the original object is not mutated.
    assert merged.debug == base.debug
    assert merged is not base
    assert base.multithread == default_settings().multithread


def test_merge_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown setting"):
        merge_settings(default_settings(), {"nope": 1})


@pytest.mark.parametrize("value", [-0.1, 1.5, "abc"])
def test_merge_rejects_out_of_range_threshold(value) -> None:
    with pytest.raises(ValueError):
        merge_settings(default_settings(), {"detection_threshold": value})


def test_merge_rejects_negative_roll() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        merge_settings(default_settings(), {"video_pre_roll_seconds": -1})


def test_merge_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="true or false"):
        merge_settings(default_settings(), {"multithread": "yes"})


def test_merge_rejects_empty_image_dir() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        merge_settings(default_settings(), {"image_dir": "   "})


def test_ignore_species_trims_and_dedupes() -> None:
    merged = merge_settings(
        default_settings(), {"ignore_species": [" Robin ", "Robin", "", "Unknown"]}
    )
    assert merged.ignore_species == ["Robin", "Unknown"]


def test_ignore_species_rejects_non_string_list() -> None:
    with pytest.raises(ValueError, match="list of strings"):
        merge_settings(default_settings(), {"ignore_species": [1, 2]})


def test_save_then_load_round_trips(tmp_path) -> None:
    path = str(tmp_path / "settings.json")
    original = merge_settings(
        default_settings(),
        {"detection_threshold": 0.33, "ignore_species": ["Unknown"], "debug": False},
    )
    save_settings(path, original)
    loaded = load_settings(path)
    assert loaded == original


def test_load_missing_file_returns_defaults(tmp_path) -> None:
    loaded = load_settings(str(tmp_path / "absent.json"))
    assert loaded == default_settings()


def test_load_corrupt_json_returns_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{ not json", encoding="utf-8")
    assert load_settings(str(path)) == default_settings()


def test_load_non_object_returns_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_settings(str(path)) == default_settings()


def test_load_invalid_value_returns_defaults(tmp_path) -> None:
    path = tmp_path / "settings.json"
    payload = {**{"detection_threshold": 5.0}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_settings(str(path)) == default_settings()


def test_settings_config_path_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SETTINGS_PATH", "/tmp/custom.json")
    assert settings_config_path() == "/tmp/custom.json"


def test_settings_config_path_default(monkeypatch) -> None:
    monkeypatch.delenv("SETTINGS_PATH", raising=False)
    assert settings_config_path() == "/data/settings.json"
