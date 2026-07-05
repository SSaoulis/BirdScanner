"""Detection-region crop domain: geometry, defaults, and persistence.

The IMX500's ``ScalerCrop`` control selects a rectangular region of the sensor's
active pixel array to feed the ISP; everything downstream (the detection stream,
the classifier ROIs, the snapshot server) sees only that region.  This module
holds the pure, camera-independent logic for that crop:

* :class:`CropRegion` — an ``(x, y, w, h)`` rectangle in **unflipped raw sensor
  pixel** coordinates (origin top-left of the sensor), plus clamping.
* conversions between sensor coordinates and the **normalized** ``[0, 1]`` box the
  UI draws over a full-sensor preview.  Although the camera applies a 180-degree
  ``vflip + hflip`` transform, libcamera applies ``ScalerCrop`` in the *same*
  orientation as the transformed preview the user sees (verified empirically:
  inverting the rotation cropped the diagonally-opposite region).  So
  :func:`normalized_to_sensor` / :func:`sensor_to_normalized` are a direct
  per-axis scale with **no** rotation — the displayed box maps to the matching
  sensor region.
* :func:`main_stream_size_for_crop` — the ISP ``main`` output size that matches a
  crop's aspect ratio (so the region is not stretched into a square stream).
* JSON load/save so the chosen region survives detector restarts.

All functions take the sensor dimensions as arguments so they can be unit-tested
without a camera.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import NamedTuple, Tuple

logger = logging.getLogger("tracking")

# Native full resolution of the Sony IMX500 sensor's active pixel array.
SENSOR_W = 4056
SENSOR_H = 3040


class NormalizedBox(NamedTuple):
    """A UI box as ``[0, 1]`` fractions of the displayed full-sensor preview.

    Attributes:
        nx: Normalized left edge.
        ny: Normalized top edge.
        nw: Normalized width.
        nh: Normalized height.
    """

    nx: float
    ny: float
    nw: float
    nh: float


class SensorDimensions(NamedTuple):
    """Sensor active-area dimensions in pixels.

    Attributes:
        w: Sensor active-area width.
        h: Sensor active-area height.
    """

    w: int = SENSOR_W
    h: int = SENSOR_H


# Smallest crop edge we allow, in sensor pixels.  Guards against degenerate
# zero/tiny regions (from a stray UI drag) that would starve the detector.
MIN_CROP_PX = 200

# Long edge of the ISP ``main`` output stream.  The crop is scaled to fit a
# rectangle whose longer side is this many pixels, preserving the crop's aspect
# ratio.  640 matches the historical square stream's edge length.
DEFAULT_LONG_SIDE = 640

# Pixel alignment for ISP output dimensions (libcamera prefers even sizes).
SIZE_ALIGN = 2


@dataclass(frozen=True)
class CropRegion:
    """A crop rectangle in unflipped raw sensor pixel coordinates.

    Attributes:
        x: Left edge of the region, in sensor pixels from the sensor's left.
        y: Top edge of the region, in sensor pixels from the sensor's top.
        w: Region width in sensor pixels.
        h: Region height in sensor pixels.
    """

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return the region as the ``(x, y, w, h)`` tuple ``ScalerCrop`` wants.

        Returns:
            The region as a plain ``(x, y, w, h)`` integer tuple.
        """
        return (self.x, self.y, self.w, self.h)

    def clamped(
        self, sensor_w: int = SENSOR_W, sensor_h: int = SENSOR_H
    ) -> "CropRegion":
        """Return a copy clamped to the sensor bounds with a minimum size.

        Width/height are first floored at :data:`MIN_CROP_PX` (and capped at the
        sensor size), then the origin is pulled in so the whole rectangle fits
        inside the sensor's active area.

        Args:
            sensor_w: Sensor active-area width in pixels.
            sensor_h: Sensor active-area height in pixels.

        Returns:
            A new :class:`CropRegion` guaranteed to lie within the sensor.
        """
        w = max(MIN_CROP_PX, min(int(self.w), sensor_w))
        h = max(MIN_CROP_PX, min(int(self.h), sensor_h))
        x = max(0, min(int(self.x), sensor_w - w))
        y = max(0, min(int(self.y), sensor_h - h))
        return CropRegion(x, y, w, h)


def default_crop_region(
    sensor_w: int = SENSOR_W, sensor_h: int = SENSOR_H
) -> CropRegion:
    """Return the historical default crop (900x900 aimed at the bird feeder).

    Mirrors the previously hardcoded ``main.py`` crop: a 900x900 square anchored
    at ``(4/13, 5/10)`` of the sensor, clamped to the sensor bounds.

    Args:
        sensor_w: Sensor active-area width in pixels.
        sensor_h: Sensor active-area height in pixels.

    Returns:
        The default :class:`CropRegion`.
    """
    crop_w = crop_h = 900
    x = int(sensor_w * (4 / 13))
    y = int(sensor_h * (5 / 10))
    return CropRegion(x, y, crop_w, crop_h).clamped(sensor_w, sensor_h)


def _clamp01(value: float) -> float:
    """Clamp a float to the closed ``[0.0, 1.0]`` interval.

    Args:
        value: The value to clamp.

    Returns:
        ``value`` constrained to ``[0.0, 1.0]``.
    """
    return max(0.0, min(1.0, value))


def normalized_to_sensor(
    box: NormalizedBox, sensor: SensorDimensions = SensorDimensions()
) -> CropRegion:
    """Convert a normalized UI box into a sensor-space :class:`CropRegion`.

    The box's ``(nx, ny)`` is its top-left corner *as displayed*. libcamera
    applies ``ScalerCrop`` in the same orientation as the (vflip+hflip)
    transformed preview, so the mapping is a direct per-axis scale — the
    displayed top-left maps straight to the ``ScalerCrop`` top-left, no rotation.

    Args:
        box: The normalized box drawn on the displayed preview.
        sensor: Sensor active-area dimensions in pixels.

    Returns:
        The corresponding sensor-space :class:`CropRegion`, clamped to bounds.
    """
    left = _clamp01(box.nx)
    top = _clamp01(box.ny)
    right = _clamp01(left + max(0.0, box.nw))
    bottom = _clamp01(top + max(0.0, box.nh))

    region = CropRegion(
        x=round(left * sensor.w),
        y=round(top * sensor.h),
        w=round((right - left) * sensor.w),
        h=round((bottom - top) * sensor.h),
    )
    return region.clamped(sensor.w, sensor.h)


def sensor_to_normalized(
    region: CropRegion, sensor: SensorDimensions = SensorDimensions()
) -> NormalizedBox:
    """Convert a sensor-space :class:`CropRegion` to a normalized UI box.

    Inverse of :func:`normalized_to_sensor`: maps a sensor region back to the
    ``(nx, ny, nw, nh)`` fractions the UI uses to position the box on the
    preview.  A direct per-axis scale (no rotation), matching how libcamera
    applies ``ScalerCrop`` relative to the transformed preview.

    Args:
        region: The sensor-space crop region.
        sensor: Sensor active-area dimensions in pixels.

    Returns:
        A :class:`NormalizedBox` of fractions in ``[0, 1]`` for the displayed
        preview.
    """
    return NormalizedBox(
        nx=region.x / sensor.w,
        ny=region.y / sensor.h,
        nw=region.w / sensor.w,
        nh=region.h / sensor.h,
    )


def _align(value: int, align: int = SIZE_ALIGN) -> int:
    """Round ``value`` to the nearest positive multiple of ``align``.

    Args:
        value: The value to align.
        align: The alignment step (must be positive).

    Returns:
        ``value`` rounded to the nearest multiple of ``align``, at least
        ``align``.
    """
    rounded = int(round(value / align)) * align
    return max(align, rounded)


def main_stream_size_for_crop(
    crop_w: int, crop_h: int, long_side: int = DEFAULT_LONG_SIDE
) -> Tuple[int, int]:
    """Return the ISP ``main`` output size matching a crop's aspect ratio.

    The ISP stretches the crop region to fill the ``main`` stream, so a square
    stream would distort a non-square crop.  This sizes the stream so its longer
    edge is ``long_side`` and its aspect ratio matches the crop, keeping pixels
    square.  Both dimensions are aligned to :data:`SIZE_ALIGN`.

    Args:
        crop_w: Crop width in sensor pixels.
        crop_h: Crop height in sensor pixels.
        long_side: Desired length of the stream's longer edge.

    Returns:
        The ``(width, height)`` for the ``main`` stream.
    """
    if crop_w <= 0 or crop_h <= 0:
        return (_align(long_side), _align(long_side))
    if crop_w >= crop_h:
        w = float(long_side)
        h = long_side * crop_h / crop_w
    else:
        h = float(long_side)
        w = long_side * crop_w / crop_h
    return (_align(int(round(w))), _align(int(round(h))))


def crop_config_path() -> str:
    """Return the path of the crop-region JSON file.

    Reads ``CROP_CONFIG_PATH``; defaults to ``/data/crop.json`` (the detector's
    writable data volume).

    Returns:
        The filesystem path to persist/read the crop region.
    """
    return os.environ.get("CROP_CONFIG_PATH", "/data/crop.json")


def load_crop_region(path: str, default: CropRegion) -> CropRegion:
    """Load a persisted crop region, falling back to ``default`` on any problem.

    A missing file, malformed JSON, or missing/invalid keys all yield
    ``default`` (clamped) rather than an error, so a corrupt config can never
    stop the detector from starting.

    Args:
        path: Path to the JSON file written by :func:`save_crop_region`.
        default: Region to return when the file cannot be read or parsed.

    Returns:
        The loaded :class:`CropRegion`, clamped to the sensor, or ``default``.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        region = CropRegion(
            x=int(data["x"]), y=int(data["y"]), w=int(data["w"]), h=int(data["h"])
        )
        return region.clamped()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Could not load crop config %s (%s); using default", path, exc)
        return default.clamped()


def save_crop_region(path: str, region: CropRegion) -> None:
    """Persist a crop region to JSON atomically.

    Writes to a temporary sibling file and renames it into place so a concurrent
    reader never sees a half-written file.  Failures are logged and swallowed:
    persistence is best-effort and must not break a live crop update.

    Args:
        path: Destination JSON path.
        region: The region to persist.
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"x": region.x, "y": region.y, "w": region.w, "h": region.h}, fh)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("Could not save crop config %s (%s)", path, exc)
