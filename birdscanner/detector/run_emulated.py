"""Run the full detector off-Pi against the emulated IMX500 camera.

This is the development entry point.  It installs the fake ``picamera2`` /
``libcamera`` modules (backed by a real YOLO11n ONNX object detector) **before**
importing :mod:`birdscanner.detector.main`, then hands off to the real ``main``.
Because only the hardware layer is faked, everything else runs for real: the
ConvNeXt classifier, the geomodel prior, the SQLite writer, the control server,
and clip encoding.  Point the API at the same ``DB_PATH`` / ``IMAGE_DIR`` to watch
the emulated detections appear in the UI.

Configuration (env vars):

* ``FAKE_CAMERA_SOURCE`` — path to a video file to read frames from; unset cycles
  the bundled ``tests/_test_images`` stills.
* ``YOLO_ONNX_PATH`` — path to the YOLO11n ONNX (default
  ``assets/models/yolo11n.onnx``).  Obtain via ``yolo export model=yolo11n.pt
  format=onnx`` — there is no runtime ``ultralytics`` dependency.
* ``FAKE_CAMERA_MAX_FRAMES`` — stop after N frames (unset runs until the source
  is exhausted / interrupted).

Usage::

    FAKE_CAMERA_SOURCE=feeder_clip.mp4 python -m birdscanner.detector.run_emulated
"""

import logging
import os
import sys
from typing import Optional

from birdscanner.detector.emulation.install import (
    build_default_detector,
    build_frame_source_from_env,
    install_fake_camera_modules,
)
from birdscanner.detector.paths import model_dir

logger = logging.getLogger("tracking")

_DEFAULT_YOLO_FILENAME = "yolo11n.onnx"


def _yolo_model_path() -> str:
    """Return the YOLO11n ONNX path from the env or the default model dir."""
    override = os.environ.get("YOLO_ONNX_PATH")
    if override:
        return override
    return str(model_dir() / _DEFAULT_YOLO_FILENAME)


def _max_frames() -> Optional[int]:
    """Return the optional ``FAKE_CAMERA_MAX_FRAMES`` cap, or ``None``."""
    raw = os.environ.get("FAKE_CAMERA_MAX_FRAMES")
    return int(raw) if raw else None


def main() -> None:
    """Install the emulated camera, then run the real detector ``main``."""
    logging.basicConfig(level=logging.INFO)

    model_path = _yolo_model_path()
    if not os.path.exists(model_path):
        logger.error(
            "YOLO ONNX model not found at %s. Export one with "
            "`yolo export model=yolo11n.pt format=onnx` and place it there, or "
            "set YOLO_ONNX_PATH.",
            model_path,
        )
        sys.exit(1)

    frame_source = build_frame_source_from_env(os.environ.get("FAKE_CAMERA_SOURCE"))
    detector = build_default_detector(model_path)
    install_fake_camera_modules(frame_source, detector, max_frames=_max_frames())

    # Import only *after* the fakes are registered, so the real camera module's
    # top-level `import libcamera` / `picamera2` resolves to the stand-ins.
    from birdscanner.detector.main import main as detector_main

    logger.info(
        "Starting emulated detector (source=%s)",
        os.environ.get("FAKE_CAMERA_SOURCE") or "test images",
    )
    detector_main()


if __name__ == "__main__":
    main()
