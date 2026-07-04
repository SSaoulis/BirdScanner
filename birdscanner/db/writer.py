"""Asynchronous writer that persists detection records to the database.

``DetectionWriter`` owns a background thread and a bounded queue so that
callers (including the camera callback thread) are never blocked by I/O.
"""

import logging
import threading
from datetime import datetime
from queue import Empty, Full, Queue
from typing import Optional

from birdscanner.db.database import SessionFactory
from birdscanner.db.models import DetectionRecord

logger = logging.getLogger(__name__)

_SENTINEL = object()
_QUEUE_MAXSIZE = 64


class DetectionWriter:
    """Writes ``DetectionRecord`` rows to SQLite from a background thread.

    The public ``write()`` method enqueues a record and returns immediately.
    A dedicated daemon thread drains the queue and commits each row.  If the
    queue is full the record is silently dropped so the calling thread is
    never stalled.

    Usage::

        writer = DetectionWriter(session_factory)
        writer.write(timestamp=..., species=..., ...)
        # on shutdown:
        writer.stop()
    """

    def __init__(self, session_factory: SessionFactory) -> None:
        """Initialise the writer and start the background drain thread.

        Args:
            session_factory: Zero-argument callable that returns a
                ``Session`` context manager (see ``db.database.make_session_factory``).
        """
        self._session_factory = session_factory
        self._queue: Queue = Queue(maxsize=_QUEUE_MAXSIZE)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="DetectionWriter"
        )
        self._thread.start()

    def write(  # pylint: disable=too-many-arguments
        # The parameters mirror the DetectionRecord columns one-to-one; they are
        # keyword-only, so the count never hurts call-site readability.
        self,
        *,
        timestamp: datetime,
        species: str,
        confidence: float,
        image_path: str,
        thumbnail_path: str,
        detection_confidence: Optional[float] = None,
        track_id: Optional[int] = None,
        stable_frames: Optional[int] = None,
        duration_sec: Optional[float] = None,
        box_x: Optional[float] = None,
        box_y: Optional[float] = None,
        box_w: Optional[float] = None,
        box_h: Optional[float] = None,
    ) -> None:
        """Enqueue a detection record for asynchronous persistence.

        Returns immediately; if the queue is full the record is dropped.

        Args:
            timestamp: Wall-clock time of the detection.
            species: Classified species name.
            confidence: Species-classification confidence in [0, 1].
            image_path: Path to saved image, relative to IMAGE_DIR.
            thumbnail_path: Path to thumbnail, relative to IMAGE_DIR.
            detection_confidence: Object-detection (YOLO) confidence in [0, 1] (optional).
            track_id: Stable-tracker track identifier (optional).
            stable_frames: Consecutive stable frames before classification (optional).
            duration_sec: Approximate track lifetime in seconds (optional).
            box_x: Detection box left edge as a fraction [0, 1] of image width (optional).
            box_y: Detection box top edge as a fraction [0, 1] of image height (optional).
            box_w: Detection box width as a fraction [0, 1] of image width (optional).
            box_h: Detection box height as a fraction [0, 1] of image height (optional).
        """
        record = DetectionRecord(
            timestamp=timestamp,
            species=species,
            confidence=confidence,
            detection_confidence=detection_confidence,
            image_path=image_path,
            thumbnail_path=thumbnail_path,
            track_id=track_id,
            stable_frames=stable_frames,
            duration_sec=duration_sec,
            box_x=box_x,
            box_y=box_y,
            box_w=box_w,
            box_h=box_h,
        )
        try:
            self._queue.put_nowait(record)
        except Full:
            logger.warning(
                "DetectionWriter queue full; dropping detection for %s", species
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the background thread to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for the thread to exit.
        """
        self._stop_event.set()
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=timeout)

    def _drain_loop(self) -> None:
        """Background thread: commit queued records until the sentinel is received.

        Draining continues until the sentinel arrives, so stop() is guaranteed to
        flush all records that were enqueued before it was called.
        """
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                # No items yet; if we've been asked to stop there's nothing left.
                if self._stop_event.is_set():
                    break
                continue

            if item is _SENTINEL:
                self._queue.task_done()
                break

            self._commit(item)
            self._queue.task_done()

    def _commit(self, record: DetectionRecord) -> None:
        """Write a single record to the database.

        Args:
            record: The ``DetectionRecord`` to persist.
        """
        try:
            with self._session_factory() as session:
                session.add(record)
                session.commit()
        except Exception:
            logger.exception(
                "Failed to write detection record for species=%s", record.species
            )
