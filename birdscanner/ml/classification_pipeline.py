"""Classification orchestration for the detection pipeline.

Wires detections coming out of the IMX500 / tracker into the ConvNeXt species
classifier, draws annotated frames, persists high-confidence results, and
manages synchronous vs. background-thread dispatch (``ClassificationManager``).

The per-detection processing dependencies (classifier, tracker, DB writer,
best-frame selector, video callables) are bundled into a single
:class:`PipelineContext` so they travel as one object instead of a long
parameter list.
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional

import cv2
import numpy as np

from birdscanner.db.models import DetectionRecord
from birdscanner.ml.best_frame import BestFrameSelector
from birdscanner.ml.classification import (
    Classifier,
    ONNXClassifier,
    build_preprocessing,
)
from birdscanner.ml.detection_utils import (
    draw_boxes,
    iou,
    label_for_category,
    normalized_box,
    preprocess_roi,
    save_thumbnail,
)
from birdscanner.ml.tracking import (
    StableDetectionTracker,
    should_run_bird_classification_for_detection,
    stable_detection_tracker,
)

if TYPE_CHECKING:
    from queue import Queue

    from birdscanner.db.writer import DetectionWriter

# Root directory for saved images; overridable via IMAGE_DIR environment variable.
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/home/stefan/Pictures/bird_detections")

# Minimum classification confidence before a detection is saved/persisted.
_SAVE_CONFIDENCE_THRESHOLD = 0.4

logger = logging.getLogger("tracking")


# Global classification state shared with the live frame loop.
classification_results: dict[int, tuple[str | None, float | None]] = {}
# List of (box, species, confidence) tuples for temporal filtering.
last_detection_classifications: list[tuple] = []

ClassifyFn = Callable[[Classifier, np.ndarray], tuple]


class Still(NamedTuple):
    """A candidate frame plus the detection box within it.

    Attributes:
        frame: The RGB frame to classify / save.
        box: The detection box ``(x, y, w, h)`` in ``frame`` pixel coordinates.
    """

    frame: np.ndarray
    box: tuple


class Classification(NamedTuple):
    """A species-classification result.

    Attributes:
        species: The predicted species name.
        confidence: The prediction confidence in ``[0, 1]``.
    """

    species: str
    confidence: float


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

    Args:
        classifier: Classifier instance for inference.
        image: Input bird image as numpy array.

    Returns:
        tuple: (species, confidence) where species is a string and confidence is
            a float between 0.0 and 1.0.
    """
    return classifier.classify(image)


@dataclass
class PipelineContext:
    """Injected dependencies for processing a single detection.

    Bundling these into one object keeps the per-detection functions (and
    :class:`ClassificationManager`) to a short, readable parameter list. All
    fields except ``classifier`` are optional so tests and the legacy path can
    supply only what they need.

    Attributes:
        classifier: Classifier used for species classification.
        tracker: Stability tracker gating classification (defaults to the
            module-global ``stable_detection_tracker``).
        classify_fn: Callable performing the classification (overridable in tests;
            defaults to :func:`run_bird_classification`).
        detection_writer: Optional writer persisting each saved detection.
        best_frame_selector: Optional per-track best-frame store.
        record_fn: Optional callable starting a video clip; returns ``True`` when
            recording actually began.
        video_frame_fn: Optional callable fed every clean frame for the pre-roll
            buffer.
    """

    classifier: Classifier
    tracker: StableDetectionTracker = field(default=None)  # type: ignore[assignment]
    classify_fn: ClassifyFn = field(default=None)  # type: ignore[assignment]
    detection_writer: Optional["DetectionWriter"] = None
    best_frame_selector: Optional[BestFrameSelector] = None
    record_fn: Optional[Callable[[str], bool]] = None
    video_frame_fn: Optional[Callable[[np.ndarray], None]] = None

    def __post_init__(self) -> None:
        """Fill in the module-level defaults for the tracker and classify callable."""
        if self.tracker is None:
            self.tracker = stable_detection_tracker
        if self.classify_fn is None:
            self.classify_fn = run_bird_classification


def _best_still(context: PipelineContext, track, still: Still) -> Still:
    """Return the best :class:`Still` for a track, else the trigger still.

    Args:
        context: Pipeline dependencies (holds the best-frame selector).
        track: The stable track, or ``None``.
        still: The still from the frame that triggered classification.

    Returns:
        The best still observed for the track when available, otherwise
        ``still`` unchanged.
    """
    if context.best_frame_selector is None or track is None:
        return still
    best = context.best_frame_selector.take(track.track_id)
    if best is None:
        return still
    return Still(best.frame, best.box)


def _classify_track(
    context: PipelineContext, still: Still, detection_id: int, track
) -> Optional[Classification]:
    """Classify a track's ROI and mark it classified, unless the ROI is empty.

    A degenerate (zero-area) box yields an empty ROI the classifier cannot
    process; that case returns ``None`` *without* marking the track, so a later,
    non-degenerate frame can still classify it.

    Args:
        context: Pipeline dependencies (classifier + classify callable + tracker).
        still: The frame + box to classify.
        detection_id: Detection index, for logging.
        track: The stable track, or ``None``.

    Returns:
        The :class:`Classification`, or ``None`` when the ROI is empty.
    """
    roi, _ = preprocess_roi(still.frame, still.box)
    if roi.size == 0:
        logger.warning(
            "Skipping classification for detection %s: empty ROI from box %s",
            detection_id,
            still.box,
        )
        return None
    species, confidence = context.classify_fn(context.classifier, roi)
    if track is not None:
        context.tracker.mark_classified(track.track_id, species=species)
    return Classification(species, confidence)


def _save_still_and_thumbnail(species_dir: Path, stem: str, still: Still) -> None:
    """Write the clean full still (no box drawn) and its thumbnail to disk.

    Args:
        species_dir: Directory for this species' images.
        stem: Filename stem shared by the still, thumbnail, and clip.
        still: The best frame + box (the box crops the thumbnail ROI).
    """
    cv2.imwrite(
        str(species_dir / f"{stem}.png"),
        cv2.cvtColor(still.frame, cv2.COLOR_RGB2BGR),
    )
    thumb_roi, _ = preprocess_roi(still.frame, still.box)
    save_thumbnail(thumb_roi, str(species_dir / f"{stem}_thumb.jpg"))


def _start_clip(
    context: PipelineContext, species_dir: Path, stem: str, species: str
) -> Optional[str]:
    """Start a video clip for this detection, returning its relative path.

    Args:
        context: Pipeline dependencies (holds the record callable).
        species_dir: Directory for this species' clips.
        stem: Filename stem shared by the still and clip.
        species: Species name (the relative path's directory).

    Returns:
        The clip path relative to ``IMAGE_DIR`` when recording began, else
        ``None`` (no recorder, or a single-flight-declined trigger).
    """
    if context.record_fn is None:
        return None
    started = context.record_fn(str(species_dir / f"{stem}.mp4"))
    return f"{species}/{stem}.mp4" if started else None


def _persist_detection(
    context: PipelineContext,
    still: Still,
    detection,
    track,
    result: Classification,
) -> None:
    """Save the still/thumbnail/clip and write the DB row for a classified bird.

    The saved still is the *clean* best frame (no box drawn); the box is stored
    as normalized coordinates so the UI can overlay it on demand.

    Args:
        context: Pipeline dependencies (writer + recorder).
        still: The best frame + box to save.
        detection: The triggering detection (for its YOLO confidence).
        track: The stable track, or ``None``.
        result: The species classification to persist.
    """
    ts = datetime.now()
    stem = str(ts).replace(":", "-")
    species_dir = Path(IMAGE_DIR) / result.species
    species_dir.mkdir(parents=True, exist_ok=True)

    still_path = species_dir / f"{stem}.png"
    _save_still_and_thumbnail(species_dir, stem, still)
    video_rel = _start_clip(context, species_dir, stem, result.species)

    # Species is logged here (not on the stable/deleted track lines) so it only
    # appears once a stable track is actually classified as a bird and saved.
    logger.info(
        "Bird classified: track_id=%s species=%s confidence=%.0f%%",
        track.track_id if track is not None else None,
        result.species,
        result.confidence * 100,
    )
    logger.info("Saved to %s", still_path)

    if context.detection_writer is None:
        return

    norm = normalized_box(still.box, still.frame.shape)
    context.detection_writer.write(
        DetectionRecord(
            timestamp=ts,
            species=result.species,
            confidence=result.confidence,
            detection_confidence=float(detection.conf),
            image_path=f"{result.species}/{stem}.png",
            thumbnail_path=f"{result.species}/{stem}_thumb.jpg",
            video_path=video_rel,
            track_id=track.track_id if track is not None else None,
            stable_frames=track.stable_frames if track is not None else None,
            box_x=norm[0],
            box_y=norm[1],
            box_w=norm[2],
            box_h=norm[3],
        )
    )


def process_single_detection_with_stable_tracks(
    item: tuple,
    context: PipelineContext,
    results_lock: threading.Lock,
) -> None:
    """Process a detection using multi-frame stable-track gating.

    Gates classification until the track has been stable for the tracker's
    configured number of frames, classifies the track's best observed frame, and
    persists the result when it clears the confidence threshold.

    Args:
        item: Tuple of (image, detection_id, detection, labels, classifier_class).
        context: Injected pipeline dependencies.
        results_lock: Lock guarding the shared classification-results dict.
    """
    _image, detection_id, detection, _labels, classifier_class = item

    is_bird = classifier_class.lower() == "bird"
    result: Optional[Classification] = None
    track = None
    still = Still(_image, detection.box)

    if is_bird and should_run_bird_classification_for_detection(
        detection_id, tracker=context.tracker
    ):
        track = context.tracker.track_for_detection_id(detection_id)
        still = _best_still(context, track, still)
        result = _classify_track(context, still, detection_id, track)

    if (
        is_bird
        and result is not None
        and result.species
        and result.confidence
        and result.confidence > _SAVE_CONFIDENCE_THRESHOLD
    ):
        _persist_detection(context, still, detection, track, result)

    with results_lock:
        classification_results[detection_id] = (
            (result.species, result.confidence) if result is not None else (None, None)
        )


def _reuse_classification(box: tuple) -> tuple:
    """Reuse a previous frame's classification when its box overlaps ``box``.

    Args:
        box: The current detection box.

    Returns:
        ``(species, confidence)`` from a sufficiently-overlapping previous
        detection, or ``(None, None)`` when there is no match.
    """
    for last_box, last_species, last_confidence in last_detection_classifications:
        if iou(box, last_box) > 0.6:
            return last_species, last_confidence
    return None, None


def _save_legacy_detection(image_with_boxes: np.ndarray, species: str) -> None:
    """Save an annotated legacy detection image under the species directory.

    Args:
        image_with_boxes: The annotated frame to save.
        species: Classified species name (the sub-directory).
    """
    species_dir = f"/home/stefan/Pictures/bird_detections/{species}/"
    os.makedirs(species_dir, exist_ok=True)
    output_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{species_dir}{datetime.now()}.png", output_image)


def process_single_detection(
    item: tuple,
    context: PipelineContext,
    results_lock: threading.Lock,
) -> None:
    """Process one detection with the legacy per-frame temporal-reuse logic.

    NOTE: This is the legacy path (per-frame reuse). The current multi-frame
    gating logic lives in :func:`process_single_detection_with_stable_tracks`.

    Args:
        item: Tuple of (image, detection_id, detection, labels, classifier_class).
        context: Injected pipeline dependencies.
        results_lock: Lock guarding the shared classification-results dict.
    """
    image, detection_id, detection, labels, classifier_class = item

    species, confidence = _reuse_classification(detection.box)
    roi, coords = preprocess_roi(image, detection.box)
    if species is None:
        species, confidence = context.classify_fn(context.classifier, roi)

    image_with_boxes = draw_boxes(
        image.copy(), coords, detection, labels, (species, confidence)
    )

    if (
        classifier_class.lower() == "bird"
        and species
        and confidence
        and confidence > _SAVE_CONFIDENCE_THRESHOLD
    ):
        _save_legacy_detection(image_with_boxes, species)

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
    last_results: Optional[list],
    manager: "ClassificationManager",
    labels: list,
) -> None:
    """Draw detections onto ISP output and queue bird detections for classification.

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

    context = manager.context
    with MappedArray(request, stream) as m:
        full_img = m.array.copy()

        # Feed every clean frame into the video ring buffer (cheap; no encoding
        # while idle) so a triggered clip has pre-roll footage.
        if context.video_frame_fn is not None:
            context.video_frame_fn(full_img)

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
                _observe_best_frame(context, detection_id, detection, full_img)
                manager.process(
                    (full_img, detection_id, detection, labels, classifier_class)
                )


def _observe_best_frame(
    context: PipelineContext, detection_id: int, detection, frame: np.ndarray
) -> None:
    """Offer ``frame`` to the best-frame selector for the detection's track.

    Args:
        context: Pipeline dependencies (holds the best-frame selector + tracker).
        detection_id: Detection index within the frame.
        detection: The detection (box + YOLO confidence).
        frame: The clean frame to offer as a candidate.
    """
    if context.best_frame_selector is None:
        return
    track = context.tracker.track_for_detection_id(detection_id)
    if track is not None:
        context.best_frame_selector.observe(
            track.track_id, frame, detection.box, float(detection.conf)
        )


class ClassificationManager:
    """Manages bird classification processing with optional multithreading."""

    def __init__(
        self,
        context: PipelineContext,
        *,
        use_multithreading: bool = False,
        queue_maxsize: int = 0,
        use_stable_track_gating: bool = False,
    ) -> None:
        """Initialize the ClassificationManager.

        Args:
            context: Injected pipeline dependencies (see :class:`PipelineContext`).
            use_multithreading: If True, process detections on a background thread.
            queue_maxsize: Maximum queue size for async processing (0 = unlimited).
            use_stable_track_gating: If True, gate classification on track stability.
        """
        self.context = context
        self.use_multithreading = use_multithreading
        self.use_stable_track_gating = use_stable_track_gating
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
            results_lock: Threading lock synchronizing results-dict access.
        """
        self._results_lock = results_lock

    def process(self, item: tuple) -> None:
        """Process a detection item synchronously or queue it for async processing.

        In synchronous mode the detection is processed immediately on the calling
        thread. In async mode it is queued for the background worker; if the queue
        is full the item is dropped to prevent blocking the camera callback.

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

        Exceptions from processing a single detection are logged and swallowed so
        one bad detection never takes down the pipeline: in async mode an
        unhandled exception would kill the worker thread permanently and silently
        stop all further classification; in sync mode it would crash the camera
        callback.

        Args:
            item: Detection item tuple to process.
        """
        try:
            if self.use_stable_track_gating:
                process_single_detection_with_stable_tracks(
                    item,
                    context=self.context,
                    results_lock=self._results_lock,  # type: ignore[arg-type]
                )
            else:
                process_single_detection(
                    item,
                    context=self.context,
                    results_lock=self._results_lock,  # type: ignore[arg-type]
                )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Classification failed for a detection; skipping it")

    def _worker_loop(self) -> None:
        """Worker thread loop: process queued detections until the sentinel arrives."""
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
        """Stop the worker thread gracefully (a no-op in synchronous mode)."""
        if not self.use_multithreading:
            return
        self._stop_event.set()  # type: ignore
        self._queue.put(None)  # type: ignore
        self._thread.join(timeout=5)  # type: ignore
