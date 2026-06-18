"""Classification orchestration for the detection pipeline.

Wires detections coming out of the IMX500 / tracker into the ConvNeXt species
classifier, draws annotated frames, persists high-confidence results, and
manages synchronous vs. background-thread dispatch (``ClassificationManager``).
"""

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import cv2
import numpy as np

from classification import Classifier, ONNXClassifier, build_preprocessing
from detection_utils import (
    draw_boxes,
    iou,
    label_for_category,
    normalized_box,
    preprocess_roi,
    save_thumbnail,
)
from tracking import (
    StableDetectionTracker,
    should_run_bird_classification_for_detection,
    stable_detection_tracker,
)

if TYPE_CHECKING:
    from queue import Queue

    from db.writer import DetectionWriter

# Root directory for saved images; overridable via IMAGE_DIR environment variable.
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/home/stefan/Pictures/bird_detections")

logger = logging.getLogger("tracking")


# Global classification state shared with the live frame loop.
classification_results: dict[int, tuple[str | None, float | None]] = {}
# List of (box, species, confidence) tuples for temporal filtering.
last_detection_classifications: list[tuple] = []


def setup_classifier(model_path: str, class_to_idx_path: str) -> Classifier:
    """Initialize the ONNX classifier with preprocessing.

    Creates an ONNX-based bird species classifier with standard ImageNet
    preprocessing (normalization and resizing to 384x384).

    Args:
        model_path: Path to the ONNX model file.
        class_to_idx_path: Path to the class-to-index mapping JSON file.

    Returns:
        Classifier: Configured classifier instance ready for inference.
    """
    onnx_model = ONNXClassifier(str(model_path))
    preprocessing = build_preprocessing(
        {
            "size": (384, 384),
            "rgb_values": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "center_crop": 1.0,
            "simple_crop": False,
        }
    )
    return Classifier(onnx_model, class_to_idx_path, preprocessing=preprocessing)


def run_bird_classification(classifier: Classifier, image: np.ndarray) -> tuple:
    """Run bird classification on the given image.

    Performs species classification on a bird image using the configured
    ONNX classifier.

    Args:
        classifier: Classifier instance for inference.
        image: Input bird image as numpy array.

    Returns:
        tuple: (species, confidence) where species is a string and
            confidence is a float between 0.0 and 1.0.
    """
    return classifier.classify(image)


def process_single_detection_with_stable_tracks(
    item: tuple,
    *,
    results_lock: threading.Lock,
    classifier: Classifier,
    tracker: StableDetectionTracker,
    classify_fn: Optional[Callable[[Classifier, np.ndarray], tuple]] = None,
    detection_writer: Optional["DetectionWriter"] = None,
) -> None:
    """Process detection using the new multi-frame stable-track gating logic.

    The existing, older per-frame cache logic is intentionally left in
    `process_single_detection` for reference.

    Args:
        item: Tuple of (image, detection_id, detection, labels, classifier_class).
        results_lock: Lock for thread-safe access to the classification results dict.
        classifier: Classifier instance for bird species classification.
        tracker: StableDetectionTracker to gate classification by track stability.
        classify_fn: Optional override for the classification callable (used in tests).
        detection_writer: Optional DetectionWriter for persisting records to SQLite.
    """

    if classify_fn is None:
        classify_fn = run_bird_classification

    # ``labels`` is part of the queued tuple but no longer used here now that the
    # bounding box is not drawn onto the saved image.
    image, detection_id, detection, _labels, classifier_class = item

    species = None
    confidence = None
    track = None

    # Gate classification until stable over N frames.
    if (
        classifier_class.lower() == "bird"
        and should_run_bird_classification_for_detection(detection_id, tracker=tracker)
    ):
        roi, _ = preprocess_roi(image, detection.box)
        if roi.size == 0:
            # A degenerate (zero-area) detection box yields an empty ROI that the
            # classifier cannot process (PIL raises on an empty array). Skip it
            # without marking the track classified, so a later, non-degenerate
            # frame for the same track can still be classified.
            logger.warning(
                "Skipping classification for detection %s: empty ROI from box %s",
                detection_id,
                detection.box,
            )
        else:
            species, confidence = classify_fn(classifier, roi)

            track = tracker.track_for_detection_id(detection_id)
            if track is not None:
                tracker.mark_classified(track.track_id, species=species)

    # The ROI is needed for the saved thumbnail. The bounding box is no longer
    # burned into the saved full image — the raw frame is kept clean and the box
    # is persisted as normalized coordinates so the UI can overlay it on demand.
    roi, _ = preprocess_roi(image, detection.box)

    # Save only after a classification actually happened.
    if (
        classifier_class.lower() == "bird"
        and species
        and confidence
        and confidence > 0.4
    ):
        ts = datetime.now()
        species_dir = Path(IMAGE_DIR) / species
        species_dir.mkdir(parents=True, exist_ok=True)

        stem = str(ts).replace(":", "-")
        image_rel = f"{species}/{stem}.png"
        thumb_rel = f"{species}/{stem}_thumb.jpg"

        output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(species_dir / f"{stem}.png"), output_image)

        save_thumbnail(roi, str(species_dir / f"{stem}_thumb.jpg"))

        box = normalized_box(detection.box, image.shape)
        if detection_writer is not None:
            detection_writer.write(
                timestamp=ts,
                species=species,
                confidence=confidence,
                detection_confidence=float(detection.conf),
                image_path=image_rel,
                thumbnail_path=thumb_rel,
                track_id=track.track_id if track is not None else None,
                stable_frames=track.stable_frames if track is not None else None,
                box_x=box[0],
                box_y=box[1],
                box_w=box[2],
                box_h=box[3],
            )

    with results_lock:
        classification_results[detection_id] = (species, confidence)


def process_single_detection(
    item: tuple,
    *,
    results_lock: threading.Lock,
    classifier: Classifier,
) -> None:
    """Process one detection item and optionally save high-confidence results.

    Performs bird species classification on a detection, applies temporal
    filtering to reuse previous classifications when boxes overlap, and
    saves high-confidence detections to disk.

    NOTE: This is the legacy logic (per-frame reuse). The new multi-frame
    gating logic lives in `process_single_detection_with_stable_tracks`.

    Args:
        item: Tuple of (image, detection_id, detection, labels, classifier_class).
        results_lock: Thread lock for safe results dictionary access.
        classifier: Classifier instance for bird species classification.
    """
    image, detection_id, detection, labels, classifier_class = item

    # Temporal filtering: reuse classification if the box overlaps significantly
    # with any detection from the previous frame.
    species = None
    confidence = None

    for last_box, last_species, last_confidence in last_detection_classifications:
        if iou(detection.box, last_box) > 0.6:
            # Reuse classification from previous frame
            species = last_species
            confidence = last_confidence
            break

    # Run classification only if we didn't reuse
    if species is None:
        roi, coords = preprocess_roi(image, detection.box)
        species, confidence = run_bird_classification(classifier, roi)

    roi, coords = preprocess_roi(image, detection.box)
    image_with_boxes = draw_boxes(
        image.copy(), coords, detection, labels, species, confidence
    )

    if classifier_class.lower() == "bird" and species and confidence:
        if confidence > 0.4:
            time = datetime.now()
            os.makedirs(
                f"/home/stefan/Pictures/bird_detections/{species}/", exist_ok=True
            )
            output_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                f"/home/stefan/Pictures/bird_detections/{species}/{time}.png",
                output_image,
            )

    with results_lock:
        classification_results[detection_id] = (species, confidence)


def update_detection_classifications_cache(
    detections: list,
    results: dict,
) -> None:
    """Update the cache of detection classifications for the current frame.

    Builds a list of (box, species, confidence) tuples from the current
    detections and their classification results, replacing the previous
    frame's cache.

    Args:
        detections: List of Detection objects from current frame.
        results: Dictionary mapping detection_id to (species, confidence).
    """
    global last_detection_classifications

    new_classifications = []
    for detection_id, detection in enumerate(detections):
        if detection_id in results:
            species, confidence = results[detection_id]
            if species and confidence:
                new_classifications.append((detection.box, species, confidence))

    last_detection_classifications = new_classifications


def process_detections(
    request,
    stream: str,
    last_results: list,
    manager: "ClassificationManager",
    labels: list,
) -> None:
    """Draw detections onto ISP output and queue for classification.

    Processes all detections from the current frame by drawing boxes on
    the preview stream and queuing bird detections for asynchronous
    species classification.

    Args:
        request: Camera request object with frame data.
        stream: ISP output stream name (e.g., 'main').
        last_results: List of Detection objects from current frame.
        manager: ClassificationManager instance for async processing.
        labels: List of class label strings.
    """
    if last_results is None:
        return
    from picamera2 import MappedArray  # type: ignore

    with MappedArray(request, stream) as m:
        full_img = m.array.copy()

        for detection_id, detection in enumerate(last_results):
            classifier_class = label_for_category(labels, int(detection.category))
            if classifier_class is None:
                # The IMX500 SSD model occasionally emits a spurious detection
                # whose class index falls outside the label list; skip it rather
                # than crash the camera callback with an IndexError.
                logger.warning(
                    "Skipping detection with out-of-range category %s "
                    "(label count=%d)",
                    detection.category,
                    len(labels),
                )
                continue

            _, coords = preprocess_roi(full_img, detection.box)
            image_with_boxes = draw_boxes(full_img.copy(), coords, detection, labels)
            m.array[:] = image_with_boxes

            if classifier_class.lower() == "bird":
                manager.process(
                    (full_img, detection_id, detection, labels, classifier_class)
                )


class ClassificationManager:
    """Manages bird classification processing with optional multithreading."""

    def __init__(
        self,
        classifier: Classifier,
        *,
        use_multithreading: bool = False,
        queue_maxsize: int = 0,
        use_stable_track_gating: bool = False,
        tracker: Optional[StableDetectionTracker] = None,
        detection_writer: Optional["DetectionWriter"] = None,
    ) -> None:
        """Initialize the ClassificationManager.

        Creates a classification processor that can operate in synchronous or
        asynchronous mode. In async mode, detections are queued for processing
        by a background worker thread.

        Args:
            classifier: Classifier instance for bird species classification.
            use_multithreading: If True, enable async processing with background thread.
            queue_maxsize: Maximum queue size for async processing. 0 means unlimited.
            use_stable_track_gating: If True, gate classification until track is stable.
            tracker: Optional tracker instance (defaults to module-global stable_detection_tracker).
            detection_writer: Optional DetectionWriter for persisting records after each
                high-confidence classification.
        """
        self.classifier = classifier
        self.use_multithreading = use_multithreading
        self.use_stable_track_gating = use_stable_track_gating
        self.tracker = tracker or stable_detection_tracker
        self.detection_writer = detection_writer
        self._results_lock: threading.Lock | None = None
        self._queue: "Queue[tuple] | None" = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

        if self.use_multithreading:
            from queue import Queue

            self._stop_event = threading.Event()
            self._queue = Queue(maxsize=queue_maxsize)
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def set_results_lock(self, results_lock: threading.Lock) -> None:
        """Set the lock for thread-safe results access.

        Args:
            results_lock: Threading lock for synchronizing results dictionary access.
        """
        self._results_lock = results_lock

    def process(self, item: tuple) -> None:
        """Process a detection item synchronously or queue it for async processing.

        In synchronous mode, the detection is processed immediately on the
        calling thread. In async mode, it is queued for the background worker.
        If the queue is full, the item is dropped to prevent frame blocking.

        Args:
            item: Detection item tuple to process.
        """
        if not self.use_multithreading:
            self._dispatch(item)
            return

        from queue import Full

        try:
            self._queue.put_nowait(item)  # type: ignore
        except Full:
            return

    def _dispatch(self, item: tuple) -> None:
        """Run the configured per-detection processing for one item.

        Exceptions raised while processing a single detection (e.g. a degenerate
        ROI the classifier rejects, or an I/O error while saving) are logged and
        swallowed so one bad detection never takes down the pipeline. In async
        mode an unhandled exception would kill the worker thread permanently and
        silently stop all further classification; in sync mode it would crash
        the camera callback.

        Args:
            item: Detection item tuple to process.
        """
        try:
            if self.use_stable_track_gating:
                process_single_detection_with_stable_tracks(
                    item,
                    results_lock=self._results_lock,  # type: ignore[arg-type]
                    classifier=self.classifier,
                    tracker=self.tracker,
                    detection_writer=self.detection_writer,
                )
            else:
                process_single_detection(
                    item,
                    results_lock=self._results_lock,  # type: ignore[arg-type]
                    classifier=self.classifier,
                )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception(
                "Classification failed for a detection; skipping it"
            )

    def _worker_loop(self) -> None:
        """Worker thread main loop for processing queued detections.

        Continuously retrieves items from the queue and processes them
        until a None sentinel value is received, indicating shutdown.
        """
        while not self._stop_event.is_set():  # type: ignore
            item = self._queue.get()  # type: ignore
            if item is None:
                self._queue.task_done()  # type: ignore
                break

            try:
                self._dispatch(item)
            finally:
                # Always mark the item done — even if _dispatch's own logging
                # somehow raised — so the queue never wedges and stop() can join.
                self._queue.task_done()  # type: ignore

    def stop(self) -> None:
        """Stop the worker thread gracefully.

        Signals the worker thread to stop and waits for it to finish
        with a 5-second timeout. In synchronous mode, this is a no-op.
        """
        if not self.use_multithreading:
            return
        self._stop_event.set()  # type: ignore
        self._queue.put(None)  # type: ignore
        self._thread.join(timeout=5)  # type: ignore
