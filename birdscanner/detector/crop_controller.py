"""Live detection-region control on the running camera.

:class:`CropController` owns the authoritative, in-memory crop region and is the
single place that mutates the camera's ``ScalerCrop`` (and, when the aspect ratio
changes, the ``main`` stream geometry).  It serialises every camera access behind
:attr:`camera_lock` so the detector's main capture loop and the snapshot server's
HTTP handler threads never drive libcamera concurrently.

Two kinds of update:

* **Pan/zoom at the same aspect ratio** — applied live via
  ``picam2.set_controls({"ScalerCrop": ...})`` with no interruption.
* **Aspect-ratio change** — requires resizing the ``main`` stream to keep pixels
  square, which means a brief ``stop() -> configure() -> start()`` cycle.

The camera-dependent calls are intentionally thin; the geometry and persistence
they rely on live in :mod:`crop` and are unit-tested without a camera.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np

from birdscanner.detector.crop import (
    CropRegion,
    default_crop_region,
    main_stream_size_for_crop,
    normalized_to_sensor,
    save_crop_region,
    sensor_to_normalized,
)

logger = logging.getLogger("tracking")

# How long to wait for a freshly-set ScalerCrop to take effect before capturing
# the full-sensor preview, and how many frames to pull while waiting.
_PREVIEW_SETTLE_TIMEOUT_S = 2.0
_PREVIEW_MAX_FRAMES = 12

# A configuration factory: given the desired ``main`` size and ScalerCrop tuple,
# build a complete picamera2 configuration object.  Supplied by ``main.py`` so
# all picamera2-specific knobs (format, buffer count, transform, frame rate)
# stay in one place.
ConfigFactory = Callable[[Tuple[int, int], Tuple[int, int, int, int]], Any]


class CropController:
    """Applies and persists the detection crop region on a running camera."""

    def __init__(
        self,
        picam2: Any,
        region: CropRegion,
        main_size: Tuple[int, int],
        config_factory: ConfigFactory,
        config_path: str,
        sensor_w: int,
        sensor_h: int,
    ) -> None:
        """Initialise the controller around an already-started camera.

        Args:
            picam2: The started ``Picamera2`` instance to control.
            region: The crop region the camera was started with.
            main_size: The ``(w, h)`` of the ``main`` stream the camera was
                started with (used to decide whether a reconfigure is needed).
            config_factory: Callable building a picamera2 config from a
                ``main`` size and a ScalerCrop tuple (see :data:`ConfigFactory`).
            config_path: Path to persist the region to on every change.
            sensor_w: Sensor active-area width in pixels.
            sensor_h: Sensor active-area height in pixels.
        """
        self._picam2 = picam2
        self._region = region
        self._main_size = main_size
        self._config_factory = config_factory
        self._config_path = config_path
        self._sensor_w = sensor_w
        self._sensor_h = sensor_h
        # Re-entrant so capture_full_preview can hold the lock while calling
        # other guarded helpers on the same thread.
        self.camera_lock = threading.RLock()

    def get_state(self) -> Dict[str, Any]:
        """Return the current crop as both sensor pixels and a normalized box.

        The normalized box is expressed in the displayed (180-degree-flipped)
        preview space the UI renders, so the frontend can position its overlay
        directly from it.

        Returns:
            A JSON-serialisable dict with ``x/y/w/h`` (sensor pixels), a ``norm``
            sub-dict (``nx/ny/nw/nh`` in ``[0, 1]``), and ``sensor_w/sensor_h``.
        """
        with self.camera_lock:
            region = self._region
        nx, ny, nw, nh = sensor_to_normalized(region, self._sensor_w, self._sensor_h)
        return {
            "x": region.x,
            "y": region.y,
            "w": region.w,
            "h": region.h,
            "norm": {"nx": nx, "ny": ny, "nw": nw, "nh": nh},
            "sensor_w": self._sensor_w,
            "sensor_h": self._sensor_h,
        }

    def set_from_normalized(
        self, nx: float, ny: float, nw: float, nh: float
    ) -> Dict[str, Any]:
        """Apply a crop from a normalized UI box and persist it.

        Args:
            nx: Normalized left edge of the box on the displayed preview.
            ny: Normalized top edge of the box on the displayed preview.
            nw: Normalized box width.
            nh: Normalized box height.

        Returns:
            The new state dict (see :meth:`get_state`).
        """
        region = normalized_to_sensor(nx, ny, nw, nh, self._sensor_w, self._sensor_h)
        return self._apply(region)

    def reset_to_default(self) -> Dict[str, Any]:
        """Reset the crop to the built-in default region and persist it.

        Returns:
            The new state dict (see :meth:`get_state`).
        """
        region = default_crop_region(self._sensor_w, self._sensor_h)
        return self._apply(region)

    def _apply(self, region: CropRegion) -> Dict[str, Any]:
        """Apply ``region`` to the camera (live or via reconfigure) and persist.

        Args:
            region: The new sensor-space crop region.

        Returns:
            The new state dict (see :meth:`get_state`).
        """
        new_main = main_stream_size_for_crop(region.w, region.h)
        with self.camera_lock:
            if new_main == self._main_size:
                # Same stream geometry -> a pure pan/zoom; apply live.
                self._picam2.set_controls({"ScalerCrop": region.as_tuple()})
            else:
                # Aspect ratio changed -> the main stream must be resized, which
                # needs a stop/reconfigure/start cycle.
                logger.info(
                    "Crop aspect changed; reconfiguring main stream %s -> %s",
                    self._main_size,
                    new_main,
                )
                config = self._config_factory(new_main, region.as_tuple())
                self._picam2.stop()
                self._picam2.configure(config)
                self._picam2.start()
                self._main_size = new_main
            self._region = region
            save_crop_region(self._config_path, region)
        logger.info("Detection crop set to %s", region.as_tuple())
        return self.get_state()

    def capture_full_preview_array(self) -> np.ndarray:
        """Capture one full-sensor frame for the crop editor.

        Temporarily widens ``ScalerCrop`` to the entire sensor so the user can
        see the whole scene and reposition the box, captures a settled frame from
        the ``main`` stream, then restores the previous crop.  The brief widening
        causes a momentary glitch in the live detection feed, which is acceptable
        while explicitly configuring the region.

        The returned array spans the full sensor stretched into the ``main``
        stream's current pixel grid; the frontend renders it at the true sensor
        aspect ratio so the normalized overlay maps linearly.

        Returns:
            An RGB ``numpy`` array of the full-sensor frame.
        """
        full = (0, 0, self._sensor_w, self._sensor_h)
        with self.camera_lock:
            previous = self._region.as_tuple()
            self._picam2.set_controls({"ScalerCrop": full})
            try:
                return self._capture_settled(full)
            finally:
                self._picam2.set_controls({"ScalerCrop": previous})

    def _capture_settled(self, target: Tuple[int, int, int, int]) -> np.ndarray:
        """Pull frames until the requested ScalerCrop has taken effect.

        ``set_controls`` applies on a future frame, so this discards frames until
        the per-request metadata reports the ``target`` crop (or a timeout), then
        returns that frame's ``main`` array.  Falls back to the last frame seen.

        Args:
            target: The ScalerCrop tuple expected once the control has applied.

        Returns:
            The RGB ``numpy`` array of the settled frame.
        """
        deadline = time.monotonic() + _PREVIEW_SETTLE_TIMEOUT_S
        last: np.ndarray | None = None
        for _ in range(_PREVIEW_MAX_FRAMES):
            request = self._picam2.capture_request()
            try:
                metadata = request.get_metadata()
                last = request.make_array("main")
            finally:
                request.release()
            scaler = metadata.get("ScalerCrop")
            if scaler is not None and tuple(scaler) == tuple(target):
                return last
            if time.monotonic() > deadline:
                break
        if last is None:
            raise RuntimeError("Camera produced no frame for the crop preview")
        return last
