"""Install the fake ``picamera2`` / ``libcamera`` modules for off-Pi runs.

The real :mod:`birdscanner.detector.camera` imports ``libcamera`` / ``picamera2``
at module load, so those fakes must be registered in ``sys.modules`` **before**
``birdscanner.detector.camera`` (and therefore ``birdscanner.detector.main``) is
imported.  :func:`install_fake_camera_modules` does that and wires the chosen
frame source + detector into the shared emulation state; callers then import and
run the real detector.

This is dev/test-only glue: it imports nothing Pi-only and never runs on the Pi
(there the real bindings are present and used directly).
"""

import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

from birdscanner.detector.emulation.fakes import (
    FakeIMX500,
    FakeMappedArray,
    FakeNetworkIntrinsics,
    FakePicam2,
    FakeSensorFormat,
    FakeTransform,
    active_imx500,
    build_emulation_state,
    set_emulation_state,
)
from birdscanner.detector.emulation.frames import (
    FrameSource,
    TestImagesSource,
    VideoSource,
)
from birdscanner.detector.emulation.yolo import Detector, OnnxYoloDetector

# The sys.modules keys the fakes occupy, so they can be cleanly removed again.
_FAKE_MODULE_NAMES = (
    "libcamera",
    "picamera2",
    "picamera2.formats",
    "picamera2.devices",
    "picamera2.devices.imx500",
    "picamera2.sensor_format",
)


def _make_module(name: str, **attrs: object) -> ModuleType:
    """Build a module object named ``name`` with the given attributes."""
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _imx500_factory(_model_path: str) -> FakeIMX500:
    """Return the shared fake IMX500 (mirrors ``IMX500(model_path)``)."""
    return active_imx500()


def _build_fake_modules() -> Dict[str, ModuleType]:
    """Construct the fake ``libcamera`` / ``picamera2`` module tree.

    Returns:
        A ``{module_name: module}`` mapping ready to register in ``sys.modules``
        (parents carry their submodules as attributes, so ``from
        picamera2.devices import IMX500`` resolves).
    """
    libcamera = _make_module(
        "libcamera",
        Transform=FakeTransform,
        StreamRole=_make_module("libcamera.StreamRole", Raw=object()),
    )

    formats = _make_module("picamera2.formats", is_raw=lambda _name: False)
    devices_imx500 = _make_module(
        "picamera2.devices.imx500", NetworkIntrinsics=FakeNetworkIntrinsics
    )
    devices = _make_module("picamera2.devices", IMX500=_imx500_factory)
    devices.imx500 = devices_imx500  # type: ignore[attr-defined]
    sensor_format = _make_module(
        "picamera2.sensor_format", SensorFormat=FakeSensorFormat
    )
    picamera2 = _make_module(
        "picamera2", Picamera2=FakePicam2, MappedArray=FakeMappedArray
    )
    picamera2.formats = formats  # type: ignore[attr-defined]
    picamera2.devices = devices  # type: ignore[attr-defined]
    picamera2.sensor_format = sensor_format  # type: ignore[attr-defined]

    return {
        "libcamera": libcamera,
        "picamera2": picamera2,
        "picamera2.formats": formats,
        "picamera2.devices": devices,
        "picamera2.devices.imx500": devices_imx500,
        "picamera2.sensor_format": sensor_format,
    }


def install_fake_camera_modules(
    frame_source: FrameSource,
    detector: Detector,
    *,
    max_frames: Optional[int] = None,
) -> None:
    """Register the fakes in ``sys.modules`` and install the emulation state.

    Call this **before** importing ``birdscanner.detector.camera`` /
    ``birdscanner.detector.main``.

    Args:
        frame_source: Where the emulated camera pulls frames from.
        detector: The off-Pi object detector run per frame.
        max_frames: Optional cap on frames before the capture loop unwinds
            (bounds tests / a finite dev run).
    """
    build_emulation_state(frame_source, detector, max_frames=max_frames)
    for name, module in _build_fake_modules().items():
        sys.modules[name] = module


def uninstall_fake_camera_modules() -> None:
    """Remove the fakes from ``sys.modules`` and clear the emulation state.

    Note: modules that already imported the fakes keep their bound references
    (e.g. ``camera.py``'s ``Picamera2``); this only stops *new* imports from
    resolving to the fakes, which is enough for test isolation.
    """
    for name in _FAKE_MODULE_NAMES:
        sys.modules.pop(name, None)
    set_emulation_state(None)


def _test_images() -> List[Path]:
    """Return the bundled ``tests/_test_images`` JPEG paths (may be empty)."""
    repo_root = Path(__file__).resolve().parents[3]
    image_dir = repo_root / "tests" / "_test_images"
    return sorted(image_dir.glob("*.jpg"))


def build_frame_source_from_env(source_spec: Optional[str]) -> FrameSource:
    """Build a frame source from an env-style spec.

    Args:
        source_spec: A path to a video file, or ``None``/empty to cycle the
            bundled test images.

    Returns:
        A ready :class:`FrameSource`.

    Raises:
        FileNotFoundError: If no test images are bundled and no video is given.
    """
    if source_spec:
        return VideoSource(source_spec, loop=True)
    images = _test_images()
    if not images:
        raise FileNotFoundError(
            "No FAKE_CAMERA_SOURCE given and no bundled test images found"
        )
    return TestImagesSource(images)


def build_default_detector(model_path: str) -> Detector:
    """Build the default off-Pi detector (the YOLO11n ONNX)."""
    return OnnxYoloDetector(model_path)
