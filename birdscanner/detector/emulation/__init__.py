"""Off-Pi emulation of the Sony IMX500 AI camera.

The IMX500 camera stack (``picamera2`` / ``libcamera`` + the on-chip ``.rpk``
object detector) only runs on the Raspberry Pi, which leaves the live camera
bring-up and capture loop impossible to run or test off-Pi.  This package
supplies drop-in fakes for the Pi-only modules plus a real off-Pi object detector
(YOLO11n via ONNX), so the *real* :mod:`birdscanner.detector.camera` and capture
loop execute unchanged on a development machine.

Public pieces:

* :func:`install_fake_camera_modules` — register the fakes in ``sys.modules``
  before importing the detector (see :mod:`birdscanner.detector.emulation.install`).
* :class:`OnnxYoloDetector` — the off-Pi object detector.
* :class:`TestImagesSource` / :class:`VideoSource` — frame sources.

Run the emulated detector with ``python -m birdscanner.detector.run_emulated``.
"""

from birdscanner.detector.emulation.frames import (
    FrameSource,
    TestImagesSource,
    VideoSource,
)
from birdscanner.detector.emulation.install import (
    build_default_detector,
    build_frame_source_from_env,
    install_fake_camera_modules,
    uninstall_fake_camera_modules,
)
from birdscanner.detector.emulation.yolo import (
    COCO_CLASSES,
    Detected,
    Detector,
    OnnxYoloDetector,
)

__all__ = [
    "COCO_CLASSES",
    "Detected",
    "Detector",
    "FrameSource",
    "OnnxYoloDetector",
    "TestImagesSource",
    "VideoSource",
    "build_default_detector",
    "build_frame_source_from_env",
    "install_fake_camera_modules",
    "uninstall_fake_camera_modules",
]
