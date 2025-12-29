

"""Threading logic for classification processing.

This module contains the ClassificationManager which handles both synchronous
and asynchronous processing of bird detections.
"""

from queue import Queue, Full
import threading


def process_single_detection(item, *, results_lock, classifier):
    """Process one detection item (sync or async depending on manager).
    
    This function should be imported from object_detection module.
    Keeping this here for reference.
    """
    pass


class ClassificationManager:
    """Manages bird classification processing with optional multithreading."""
    
    def __init__(self, classifier, *, use_multithreading: bool = False, queue_maxsize: int = 0):
        self.classifier = classifier
        self.use_multithreading = use_multithreading
        self._results_lock = None
        self._queue = None
        self._thread = None
        self._stop_event = None

        if self.use_multithreading:
            self._stop_event = threading.Event()
            self._queue = Queue(maxsize=queue_maxsize)
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def set_results_lock(self, results_lock):
        """Set the lock for thread-safe results access."""
        self._results_lock = results_lock

    def process(self, item):
        """Process a detection item synchronously or queue it for async processing."""
        if not self.use_multithreading:
            process_single_detection(item, results_lock=self._results_lock, classifier=self.classifier)
            return

        try:
            self._queue.put_nowait(item)
        except Full:
            # Drop frame if queue is full.
            return

    def _worker_loop(self):
        """Worker thread main loop for processing queued detections."""
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            process_single_detection(item, results_lock=self._results_lock, classifier=self.classifier)
            self._queue.task_done()

    def stop(self):
        """Stop the worker thread gracefully."""
        if not self.use_multithreading:
            return
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=5)
