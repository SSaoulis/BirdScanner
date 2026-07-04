"""On-demand short-clip recorder for bird detections.

The Raspberry Pi 5 has no hardware video encoder, so encoding is software (CPU)
and we avoid encoding continuously. Instead the camera callback feeds every
``main``-stream frame into a small in-RAM ring buffer (cheap — no encoding while
idle). When a detection fires, :meth:`VideoRecorder.trigger` captures the buffered
pre-roll frames, keeps collecting live frames for a post-roll window, and encodes
the whole sequence to an ``.mp4`` on a background thread so the camera callback is
never blocked.

This module is OpenCV/picamera-adjacent and therefore lives in ``detector/``; the
platform-independent ``ml/`` pipeline triggers it through an injected callable.
"""

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("tracking")


class VideoRecorder:
    """Records short clips around detections from a raw-frame RAM buffer.

    A fixed-length ``deque`` retains the most recent frames (the pre-roll). A
    trigger snapshots that buffer, gathers ``post_roll_seconds`` of subsequent
    frames, and encodes an mp4 on a worker thread. Only one clip records at a
    time (single-flight): triggers received while a clip is in progress are
    ignored, bounding CPU and memory.
    """

    def __init__(
        self,
        *,
        fps: float,
        pre_roll_seconds: float = 3.0,
        post_roll_seconds: float = 4.0,
    ) -> None:
        """Initialise the recorder.

        Args:
            fps: Frame rate used both to size the pre-roll buffer and to time the
                post-roll window / encode the output; should match the camera's
                effective frame delivery rate.
            pre_roll_seconds: Seconds of already-buffered frames to prepend.
            post_roll_seconds: Seconds of frames to keep collecting after a trigger.
        """
        self.fps = max(1.0, float(fps))
        self.pre_roll_seconds = pre_roll_seconds
        self.post_roll_seconds = post_roll_seconds

        pre_roll_frames = max(1, int(self.pre_roll_seconds * self.fps))
        self._buffer: Deque[np.ndarray] = deque(maxlen=pre_roll_frames)
        self._lock = threading.Lock()

        # Populated only while a clip is actively being collected.
        self._recording = False
        self._collecting: Optional[List[np.ndarray]] = None
        self._collect_until = 0.0
        self._dest_path: Optional[str] = None

    def add_frame(self, frame: np.ndarray) -> None:
        """Feed one frame from the camera callback into the recorder.

        Appends to the pre-roll ring buffer and, while a clip is being recorded,
        to the in-progress clip. When the post-roll window elapses the collected
        frames are handed off to a background encode thread. Cheap and
        non-blocking so it is safe to call every frame on the camera thread.

        The frame reference is stored as-is; callers must pass a frame that will
        not be mutated afterwards (the pipeline passes the per-frame ``full_img``
        copy, which nothing mutates in place).

        Args:
            frame: The full RGB ``main``-stream frame for this camera frame.
        """
        with self._lock:
            self._buffer.append(frame)
            if not self._recording:
                return

            assert self._collecting is not None
            self._collecting.append(frame)
            if time.monotonic() < self._collect_until:
                return

            # Post-roll window elapsed: detach the clip and encode off-thread.
            frames = self._collecting
            dest = self._dest_path
            self._recording = False
            self._collecting = None
            self._dest_path = None

        if dest is not None:
            threading.Thread(
                target=self._encode,
                args=(frames, dest),
                daemon=True,
                name="VideoRecorder-encode",
            ).start()

    def trigger(self, dest_path: str) -> bool:
        """Begin recording a clip to ``dest_path`` (mp4).

        Seeds the clip with the buffered pre-roll frames and starts collecting
        post-roll frames. Single-flight: returns ``False`` without starting a new
        clip if one is already being recorded.

        Args:
            dest_path: Absolute path for the output ``.mp4`` file.

        Returns:
            ``True`` if a new clip started, ``False`` if a clip was already in
            progress.
        """
        with self._lock:
            if self._recording:
                return False
            self._recording = True
            self._collecting = list(self._buffer)  # pre-roll snapshot
            self._collect_until = time.monotonic() + self.post_roll_seconds
            self._dest_path = dest_path
            return True

    def _encode(self, frames: List[np.ndarray], dest_path: str) -> None:
        """Encode collected frames to an mp4 file (runs on a worker thread).

        Args:
            frames: Ordered RGB frames to write.
            dest_path: Absolute output path for the ``.mp4``.
        """
        if not frames:
            logger.warning("VideoRecorder: no frames to encode for %s", dest_path)
            return

        height, width = frames[0].shape[:2]
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        # mp4v is available via the apt python3-opencv FFmpeg backend and yields a
        # browser-playable MP4; frames are RGB, OpenCV expects BGR (same
        # convention as the still writes in the classification pipeline).
        # cv2's members are populated dynamically, so mypy can't see this one.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(dest_path, fourcc, self.fps, (width, height))
        if not writer.isOpened():
            logger.error("VideoRecorder: failed to open writer for %s", dest_path)
            return
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        logger.info(
            "VideoRecorder: wrote %d-frame clip to %s", len(frames), dest_path
        )
