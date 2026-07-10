"""Frame sources for the emulated camera.

The emulated IMX500 pulls frames from a :class:`FrameSource` instead of the real
sensor.  Two concrete sources are provided:

* :class:`TestImagesSource` — cycles a fixed list of still images (the bundled
  ``tests/_test_images`` set), so the emulator runs with zero setup and produces
  deterministic frames for tests.
* :class:`VideoSource` — reads frames sequentially from a video file via
  ``cv2.VideoCapture``, the closest thing to a live feed for exercising tracking
  and stability over time.

Frames are returned as ``(H, W, 3)`` uint8 **RGB** arrays (the format the whole
pipeline assumes).  Only ``numpy`` / ``cv2`` / ``PIL`` are imported — nothing
Pi-only.
"""

from itertools import cycle
from pathlib import Path
from typing import Iterator, List, Optional, Protocol

import cv2
import numpy as np
from PIL import Image


class FrameSource(Protocol):
    """A source of RGB frames for the emulated camera."""

    def next_frame(self) -> Optional[np.ndarray]:
        """Return the next RGB frame, or ``None`` when the source is exhausted."""


class TestImagesSource:
    """Cycles a fixed list of still images as RGB frames.

    The images are decoded once up front and served round-robin, so the source
    never exhausts (``next_frame`` always returns a frame).  This makes it ideal
    for deterministic tests and a zero-setup dev run.
    """

    # Tell pytest this is not a test class despite the "Test" name prefix.
    __test__ = False

    def __init__(self, image_paths: List[Path]) -> None:
        """Decode the given images into RGB frames.

        Args:
            image_paths: Paths to the still images to cycle.

        Raises:
            ValueError: If no image paths are given.
        """
        if not image_paths:
            raise ValueError("TestImagesSource needs at least one image path")
        self._frames: List[np.ndarray] = [
            np.asarray(Image.open(path).convert("RGB")) for path in image_paths
        ]
        self._cycle: Iterator[np.ndarray] = cycle(self._frames)

    def next_frame(self) -> Optional[np.ndarray]:
        """Return the next image in the cycle (never ``None``)."""
        return next(self._cycle)


class VideoSource:
    """Reads frames sequentially from a video file via ``cv2.VideoCapture``.

    OpenCV decodes frames as BGR; each is converted to RGB to match the rest of
    the pipeline.  When ``loop`` is set the video restarts from the first frame
    on exhaustion instead of returning ``None``.
    """

    def __init__(self, path: str, *, loop: bool = True) -> None:
        """Open the video file.

        Args:
            path: Path to the video file (mp4/mov/...).
            loop: Restart from the beginning when the video ends.

        Raises:
            ValueError: If the video cannot be opened.
        """
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise ValueError(f"Could not open video file: {path}")
        self._loop = loop

    def next_frame(self) -> Optional[np.ndarray]:
        """Return the next video frame as RGB, or ``None`` when exhausted.

        When ``loop`` is set, rewinds and retries once at end-of-stream.
        """
        ok, frame = self._capture.read()
        if not ok:
            if not self._loop:
                return None
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if not ok:
                return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        """Release the underlying video capture."""
        self._capture.release()
