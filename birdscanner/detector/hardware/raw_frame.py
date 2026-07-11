"""Full-FOV clip frames from the camera's raw sensor stream.

The saved detection stills/thumbnails and the live preview all come from the
``main`` ISP stream, which is cropped to the ``ScalerCrop`` detection region (the
feeder crop).  The video clip, by contrast, is meant to show the *whole,
uncropped* scene.  There is only one ``ScalerCrop`` per camera and it applies to
every processed stream, so the only source of the full field of view is the
``raw`` sensor stream (the always-allocated 2028x1520 readout, unaffected by
``ScalerCrop``).

This module turns a raw Bayer frame into a downscaled RGB frame suitable for the
clip recorder:

* :func:`bayer_cv2_code` — map the configured raw format string to the OpenCV
  demosaic code (pure; unit-tested).
* :class:`RawToRgb` — unpack to 8-bit, debayer, and downscale a raw frame (pure
  numpy/OpenCV; unit-tested with synthetic Bayer arrays).
* :func:`build_clip_frame_source` — wire a live ``Picamera2`` into a
  ``request -> RGB frame`` callable the pipeline injects as
  ``PipelineContext.video_frame_source``.

Only ``cv2``/``numpy`` are imported (the ``Picamera2``/request objects are passed
in), so the conversion is testable without a camera.
"""

import logging
import re
from typing import Any, Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("tracking")

# Longer edge of the downscaled clip frame.  Matches the historical clip footprint
# (the square crop stream was 640 on a side), so the full-FOV clip is comparable in
# size and the per-frame RAM cost stays ~1 MB.
DEFAULT_CLIP_LONG_SIDE = 640

# The full-field-of-view binned sensor mode of the IMX500.  This is the readout
# that is already allocated for the raw stream, so requesting it explicitly does
# not add a new sensor mode.
FULL_FOV_RAW_SIZE = (2028, 1520)

# libcamera raw formats name the Bayer pattern of the *first* 2x2 tile (e.g.
# ``SBGGR10`` => B G / G R).  OpenCV's ``COLOR_BAYER_XY2RGB`` codes name the tile
# starting one pixel in, so the letters are swapped relative to libcamera.  This
# maps each libcamera pattern to the OpenCV code that yields RGB output.  (The
# exact pattern depends on the sensor flip state, which is why the caller derives
# it from the *configured* format rather than hard-coding one — see
# :func:`build_clip_frame_source`.)
_BAYER_TO_CV2 = {
    "BGGR": cv2.COLOR_BAYER_RG2RGB,
    "GBRG": cv2.COLOR_BAYER_GR2RGB,
    "GRBG": cv2.COLOR_BAYER_GB2RGB,
    "RGGB": cv2.COLOR_BAYER_BG2RGB,
}

_FORMAT_RE = re.compile(r"^S([RGB]{4})(\d+)")


def bayer_cv2_code(raw_format: str) -> int:
    """Return the OpenCV demosaic code for a libcamera raw format string.

    Args:
        raw_format: A libcamera raw format, e.g. ``"SBGGR10"`` or
            ``"SRGGB12_CSI2P"``. Only the Bayer pattern letters are used.

    Returns:
        The ``cv2.COLOR_BAYER_*2RGB`` code that demosaics that pattern to RGB.

    Raises:
        ValueError: If the format string has no recognisable Bayer pattern.
    """
    match = _FORMAT_RE.match(raw_format)
    if match is None or match.group(1) not in _BAYER_TO_CV2:
        raise ValueError(f"Unrecognised raw Bayer format: {raw_format!r}")
    return _BAYER_TO_CV2[match.group(1)]


def _bit_depth(raw_format: str) -> int:
    """Return the sample bit depth encoded in a raw format string.

    Args:
        raw_format: A libcamera raw format, e.g. ``"SBGGR10"``.

    Returns:
        The bit depth (e.g. ``10``), or ``8`` if none is present.
    """
    match = _FORMAT_RE.match(raw_format)
    return int(match.group(2)) if match else 8


def _even(value: int) -> int:
    """Round ``value`` down to an even number at least 2 (OpenCV prefers even sizes)."""
    return max(2, value - (value % 2))


class RawToRgb:
    """Converts raw Bayer sensor frames into downscaled RGB frames.

    The conversion is: right-shift the samples to 8-bit, demosaic to RGB with the
    OpenCV code for the sensor's Bayer pattern, then downscale so the longer edge
    is ``long_side`` while preserving the aspect ratio (so the clip is the true,
    un-stretched full field of view).
    """

    def __init__(
        self, raw_format: str, long_side: int = DEFAULT_CLIP_LONG_SIDE
    ) -> None:
        """Build a converter for a specific raw format.

        Args:
            raw_format: The libcamera raw format string of the stream to convert
                (e.g. ``"SBGGR10"``); determines the demosaic code and bit shift.
            long_side: Desired length of the output's longer edge in pixels.
        """
        self._code = bayer_cv2_code(raw_format)
        self._shift = max(0, _bit_depth(raw_format) - 8)
        self._long_side = long_side

    def convert(self, raw: np.ndarray) -> np.ndarray:
        """Convert one raw Bayer frame to a downscaled RGB frame.

        Args:
            raw: The raw sensor frame as a 2-D (or ``(H, W, 1)``) array. Samples
                are assumed right-aligned in their integer type.

        Returns:
            An ``(h, w, 3)`` uint8 RGB array of the full field of view, downscaled
            to ``long_side`` on its longer edge.
        """
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        if raw.dtype == np.uint8:
            eight = raw
        else:
            eight = np.right_shift(raw, self._shift).astype(np.uint8)
        rgb = cv2.cvtColor(eight, self._code)
        return self._downscale(rgb)

    def _downscale(self, rgb: np.ndarray) -> np.ndarray:
        """Downscale an RGB frame so its longer edge is ``long_side``.

        Args:
            rgb: The full-resolution RGB frame.

        Returns:
            The area-resampled RGB frame (unchanged if already small enough).
        """
        height, width = rgb.shape[:2]
        longest = max(width, height)
        if longest <= self._long_side:
            return rgb
        scale = self._long_side / longest
        size = (_even(int(round(width * scale))), _even(int(round(height * scale))))
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def build_clip_frame_source(
    picam2: Any, long_side: int = DEFAULT_CLIP_LONG_SIDE
) -> Callable[[Any], Optional[np.ndarray]]:
    """Build a ``request -> full-FOV RGB frame`` source for the clip recorder.

    Reads the *actual* configured raw format back from the running camera (so the
    demosaic code matches the sensor's current flip state) and returns a callable
    that extracts, debayers, and downscales the raw stream for each camera request.
    The callable never raises: any failure is logged and yields ``None`` so the
    pipeline falls back to the cropped ``main`` frame rather than crashing the
    camera callback.

    Args:
        picam2: The started ``Picamera2`` instance (must have a ``raw`` stream
            configured).
        long_side: Desired length of the output's longer edge in pixels.

    Returns:
        A callable taking a camera ``request`` and returning the full-FOV RGB
        frame, or ``None`` when the raw frame could not be converted.
    """
    raw_format = str(picam2.camera_configuration()["raw"]["format"])
    converter = RawToRgb(raw_format, long_side=long_side)
    logger.info(
        "Full-frame clip source ready: raw format=%s -> RGB (long side %d px)",
        raw_format,
        long_side,
    )

    def source(request: Any) -> Optional[np.ndarray]:
        """Return the downscaled full-FOV RGB frame for one camera request."""
        try:
            return converter.convert(request.make_array("raw"))
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Could not build full-frame clip frame from raw stream")
            return None

    return source
