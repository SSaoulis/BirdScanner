"""Tests for the raw-stream -> full-FOV RGB conversion used by the clip recorder.

The conversion is pure numpy/OpenCV (the ``Picamera2``/request objects are passed
in), so these exercise it with synthetic Bayer arrays and a fake camera — no
camera, model, or DB. The exact Bayer pattern under the sensor flip is derived
from the *configured* format at runtime; these tests pin the mapping table and
the unpack/downscale mechanics.
"""

import types

import cv2
import numpy as np
import pytest

from birdscanner.detector.raw_frame import (
    RawToRgb,
    bayer_cv2_code,
    build_clip_frame_source,
)


@pytest.mark.parametrize(
    "raw_format,expected",
    [
        ("SBGGR10", cv2.COLOR_BAYER_RG2RGB),
        ("SGBRG10", cv2.COLOR_BAYER_GR2RGB),
        ("SGRBG10", cv2.COLOR_BAYER_GB2RGB),
        ("SRGGB10", cv2.COLOR_BAYER_BG2RGB),
        ("SRGGB12_CSI2P", cv2.COLOR_BAYER_BG2RGB),  # suffix + bit depth ignored
    ],
)
def test_bayer_cv2_code_maps_each_pattern(raw_format, expected):
    """Each libcamera Bayer pattern maps to its OpenCV demosaic code."""
    assert bayer_cv2_code(raw_format) == expected


@pytest.mark.parametrize("bad", ["", "RGGB10", "SXYZW10", "SRG10"])
def test_bayer_cv2_code_rejects_unknown_format(bad):
    """An unrecognisable format string raises rather than guessing a code."""
    with pytest.raises(ValueError):
        bayer_cv2_code(bad)


def test_convert_returns_rgb_uint8():
    """A raw Bayer frame becomes an ``(H, W, 3)`` uint8 RGB frame."""
    raw = np.zeros((8, 8), dtype=np.uint8)
    rgb = RawToRgb("SRGGB8").convert(raw)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8


def test_convert_shifts_to_8_bit_by_depth():
    """10-bit samples are right-shifted by 2 so a near-max field lands at 255."""
    # A uniform Bayer field demosaics to a uniform RGB field, so 1020 >> 2 == 255.
    raw = np.full((8, 8), 1020, dtype=np.uint16)
    rgb = RawToRgb("SRGGB10").convert(raw)
    assert int(rgb.min()) == 255 and int(rgb.max()) == 255


def test_convert_squeezes_trailing_singleton_dim():
    """A ``(H, W, 1)`` raw array is accepted the same as ``(H, W)``."""
    raw = np.zeros((8, 8, 1), dtype=np.uint8)
    rgb = RawToRgb("SRGGB8").convert(raw)
    assert rgb.shape == (8, 8, 3)


def test_convert_downscales_to_long_side_preserving_aspect():
    """A full-FOV frame is downscaled so its longer edge is ``long_side``."""
    raw = np.zeros((1520, 2028), dtype=np.uint16)  # 4:3 full-FOV binned mode
    rgb = RawToRgb("SRGGB10", long_side=640).convert(raw)
    # 2028 -> 640 on the long edge; 1520 scales to 480, aspect preserved.
    assert rgb.shape == (480, 640, 3)


def test_convert_does_not_upscale_small_frames():
    """Frames already within ``long_side`` are left at their native size."""
    raw = np.zeros((8, 8), dtype=np.uint8)
    rgb = RawToRgb("SRGGB8", long_side=640).convert(raw)
    assert rgb.shape[:2] == (8, 8)


def _fake_picam2(raw_format: str) -> types.SimpleNamespace:
    """A minimal ``Picamera2`` stand-in exposing the configured raw format."""
    return types.SimpleNamespace(
        camera_configuration=lambda: {"raw": {"format": raw_format}}
    )


def test_build_clip_frame_source_converts_raw_request():
    """The built source pulls the raw array off the request and converts it."""
    raw = np.full((8, 8), 1020, dtype=np.uint16)
    request = types.SimpleNamespace(make_array=lambda _stream: raw)

    source = build_clip_frame_source(_fake_picam2("SRGGB10"), long_side=640)
    frame = source(request)

    assert frame is not None
    assert frame.shape == (8, 8, 3)
    assert int(frame.max()) == 255


def test_build_clip_frame_source_returns_none_on_failure():
    """A failure pulling/converting the raw frame yields None (never raises)."""
    request = types.SimpleNamespace(
        make_array=lambda _stream: (_ for _ in ()).throw(RuntimeError("no raw"))
    )
    source = build_clip_frame_source(_fake_picam2("SRGGB10"))
    assert source(request) is None
