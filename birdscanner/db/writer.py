"""Asynchronous writer that persists detection records to the database.

``DetectionWriter`` owns a background thread and a bounded queue so that
callers (including the camera callback thread) are never blocked by I/O.
"""

import logging
import threading
from queue import Empty, Full, Queue

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
        writer.write(DetectionRecord(timestamp=..., species=..., ...))
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

    def write(self, record: DetectionRecord) -> None:
        """Enqueue a detection record for asynchronous persistence.

        Returns immediately; if the queue is full the record is dropped. The
        caller builds the ``DetectionRecord`` (its keyword-only constructor keeps
        every column named at the call site), so the writer stays a thin
        transport with no column list of its own to maintain.

        Args:
            record: The fully-populated detection row to persist.
        """
        try:
            self._queue.put_nowait(record)
        except Full:
            logger.warning(
                "DetectionWriter queue full; dropping detection for %s",
                record.species,
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
