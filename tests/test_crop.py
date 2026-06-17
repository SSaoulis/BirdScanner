"""Tests for the pure crop geometry + persistence helpers (no camera needed)."""

import json

import pytest

from src.crop import (
    MIN_CROP_PX,
    SENSOR_H,
    SENSOR_W,
    CropRegion,
    crop_config_path,
    default_crop_region,
    load_crop_region,
    main_stream_size_for_crop,
    normalized_to_sensor,
    save_crop_region,
    sensor_to_normalized,
)


def test_default_crop_region_is_square_within_bounds() -> None:
    region = default_crop_region()
    assert (region.w, region.h) == (900, 900)
    assert 0 <= region.x <= SENSOR_W - region.w
    assert 0 <= region.y <= SENSOR_H - region.h


def test_full_normalized_box_maps_to_full_sensor() -> None:
    region = normalized_to_sensor(0.0, 0.0, 1.0, 1.0)
    assert region == CropRegion(0, 0, SENSOR_W, SENSOR_H)


def test_normalized_to_sensor_maps_directly_without_rotation() -> None:
    # libcamera applies ScalerCrop in the same orientation as the transformed
    # preview, so a box in the top-left of the display maps to the top-left of
    # the sensor crop (no 180-degree inversion).
    region = normalized_to_sensor(0.0, 0.0, 0.25, 0.25)
    assert region.x == 0
    assert region.y == 0
    assert region.w == round(0.25 * SENSOR_W)
    assert region.h == round(0.25 * SENSOR_H)


def test_bottom_right_box_maps_to_bottom_right_crop() -> None:
    # Regression: drawing the box in the bottom-right must crop the bottom-right
    # region, not the diagonally-opposite top-left.
    region = normalized_to_sensor(0.6, 0.6, 0.4, 0.4)
    assert region.x == round(0.6 * SENSOR_W)
    assert region.y == round(0.6 * SENSOR_H)


def test_normalized_round_trip() -> None:
    region = default_crop_region()
    nx, ny, nw, nh = sensor_to_normalized(region)
    back = normalized_to_sensor(nx, ny, nw, nh)
    assert back == region


def test_tiny_box_is_clamped_to_minimum_size() -> None:
    region = normalized_to_sensor(0.5, 0.5, 0.0001, 0.0001)
    assert region.w >= MIN_CROP_PX
    assert region.h >= MIN_CROP_PX


def test_clamped_keeps_region_inside_sensor() -> None:
    region = CropRegion(x=4000, y=3000, w=900, h=900).clamped()
    assert region.x + region.w <= SENSOR_W
    assert region.y + region.h <= SENSOR_H


@pytest.mark.parametrize(
    "crop_w,crop_h,expected",
    [
        (900, 900, (640, 640)),
        (1200, 600, (640, 320)),
        (600, 1200, (320, 640)),
    ],
)
def test_main_stream_size_matches_crop_aspect(crop_w, crop_h, expected) -> None:
    assert main_stream_size_for_crop(crop_w, crop_h) == expected


def test_main_stream_size_is_even_aligned() -> None:
    w, h = main_stream_size_for_crop(901, 533)
    assert w % 2 == 0 and h % 2 == 0


def test_save_then_load_round_trips(tmp_path) -> None:
    path = str(tmp_path / "crop.json")
    region = CropRegion(100, 200, 800, 600)
    save_crop_region(path, region)
    assert load_crop_region(path, default_crop_region()) == region


def test_load_missing_file_returns_default(tmp_path) -> None:
    default = default_crop_region()
    assert load_crop_region(str(tmp_path / "nope.json"), default) == default


def test_load_corrupt_file_returns_default(tmp_path) -> None:
    path = tmp_path / "crop.json"
    path.write_text("not json", encoding="utf-8")
    default = default_crop_region()
    assert load_crop_region(str(path), default) == default


def test_load_missing_keys_returns_default(tmp_path) -> None:
    path = tmp_path / "crop.json"
    path.write_text(json.dumps({"x": 1, "y": 2}), encoding="utf-8")
    default = default_crop_region()
    assert load_crop_region(str(path), default) == default


def test_crop_config_path_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CROP_CONFIG_PATH", "/tmp/custom-crop.json")
    assert crop_config_path() == "/tmp/custom-crop.json"
    monkeypatch.delenv("CROP_CONFIG_PATH", raising=False)
    assert crop_config_path() == "/data/crop.json"
