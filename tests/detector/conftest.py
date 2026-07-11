"""Shared fixtures for the detector emulation tests.

The :func:`camera_emulator` fixture installs the fake ``picamera2`` / ``libcamera``
modules (so the real camera bring-up + capture loop run off-Pi) and tears them
back out afterwards, keeping the global ``sys.modules`` mutation contained to the
tests that opt in.  It is non-autouse, so the other detector tests are unaffected.
"""

from typing import Callable, Iterator, List, Optional

import numpy as np
import pytest

from dev.emulation.frames import FrameSource
from dev.emulation.yolo import Detected, Detector


class SolidFrameSource:
    """A frame source that always yields the same solid-colour RGB frame."""

    def __init__(self, width: int = 640, height: int = 480, fill: int = 120) -> None:
        """Record the frame geometry and fill value."""
        self._frame = np.full((height, width, 3), fill, dtype=np.uint8)

    def next_frame(self) -> Optional[np.ndarray]:
        """Return the (copied) solid frame."""
        return self._frame.copy()


class FixedBoxDetector:
    """A deterministic detector that reports one fixed bird box per frame."""

    def __init__(
        self,
        box: tuple = (0.3, 0.3, 0.7, 0.7),
        score: float = 0.95,
        label: str = "bird",
    ) -> None:
        """Record the box, score, and class label to emit."""
        self._detected = Detected(box, score, label)

    def detect(self, frame: np.ndarray) -> List[Detected]:
        """Return the single fixed detection regardless of frame content."""
        del frame
        return [self._detected]


@pytest.fixture()
def camera_emulator() -> Iterator[Callable[..., None]]:
    """Return an installer for the fake camera modules; auto-uninstalls on teardown.

    The returned callable takes a :class:`FrameSource` + :class:`Detector` (plus an
    optional ``max_frames`` cap) and registers the fakes.  Import
    ``birdscanner.detector.hardware.camera`` / ``.main`` only *after* calling it.
    """
    from dev.emulation.install import (
        install_fake_camera_modules,
        uninstall_fake_camera_modules,
    )

    installed = {"done": False}

    def _install(
        frame_source: FrameSource,
        detector: Detector,
        *,
        max_frames: Optional[int] = None,
    ) -> None:
        install_fake_camera_modules(frame_source, detector, max_frames=max_frames)
        installed["done"] = True

    yield _install

    if installed["done"]:
        uninstall_fake_camera_modules()
