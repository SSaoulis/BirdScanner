"""Classification orchestration for the detection pipeline.

Wires detections coming out of the IMX500 / tracker into the ConvNeXt species
classifier, draws annotated frames, persists high-confidence results, and
manages synchronous vs. background-thread dispatch (``ClassificationManager``).

The per-detection processing dependencies (classifier, tracker, DB writer,
best-frame selector, video callables) are bundled into a single
:class:`PipelineContext` so they travel as one object instead of a long
parameter list.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional

import cv2
import numpy as np

from birdscanner.db.models import DetectionRecord
from birdscanner.ml.best_frame import BestFrameSelector
from birdscanner.ml.classification import (
    Classifier,
    ONNXClassifier,
    build_preprocessing,
    top_k_predictions,
)
from birdscanner.ml.detection_utils import (
    draw_boxes,
    label_for_category,
    normalized_box,
    preprocess_roi,
    save_thumbnail,
)
from birdscanner.ml.geomodel import week_of_year
from birdscanner.ml.tracking import (
    StableDetectionTracker,
    should_run_bird_classification_for_detection,
    stable_detection_tracker,
)

if TYPE_CHECKING:
    from queue import Queue

    from birdscanner.db.writer import DetectionWriter
    from birdscanner.ml.geomodel import GeoPriorAdjuster

# Root directory for saved images; overridable via IMAGE_DIR environment variable.
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/home/stefan/Pictures/bird_detections")

# Default minimum classification confidence before a detection is saved. Used as
# the initial value of ``PipelineContext.save_confidence_threshold``; the Settings
# page can override it live (see ``detector/settings_controller.py``).
DEFAULT_SAVE_CONFIDENCE_THRESHOLD = 0.4

# How many of the classifier's own top predictions to persist per detection
# (``classifier_scores``), for the Advanced-stats panel.
CLASSIFIER_TOP_K = 5

# Reasons a detection ends up with no video clip (persisted as ``no_video_reason``
# so the UI can explain the greyed-out Video control). ``None`` means a clip exists.
NO_VIDEO_RECORDER_BUSY = "recorder_busy"  # single-flight recorder declined the trigger
NO_VIDEO_DISABLED = "disabled"  # no recorder wired (save_video off)

logger = logging.getLogger("tracking")


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

    When the geomodel Bayesian update ran, ``species``/``confidence`` hold the
    *posterior* (geomodel-corrected) prediction, and the extra fields carry the
    classifier's own pick plus the pre-normalised scores for debugging. When it did
    not run they are ``None`` and ``species``/``confidence`` are the classifier's
    unadjusted output.

    Attributes:
        species: The predicted species name (posterior argmax when adjusted).
        confidence: The prediction confidence in ``[0, 1]``.
        classifier_species: The classifier's own top class before the update, or
            ``None`` when the geomodel update did not run.
        classifier_confidence: The classifier's confidence for
            ``classifier_species``, or ``None``.
        geo_scores: JSON of the top pre-normalised ``[species, score]`` pairs, or
            ``None`` when the update did not run.
        classifier_scores: JSON of the classifier's own top-k ``[species,
            probability]`` softmax pairs (the raw distribution before any geomodel
            reweighting), or ``None`` when the classifier could not produce a full
            distribution (e.g. a test fake without ``predict_proba``).
    """

    species: str
    confidence: float
    classifier_species: Optional[str] = None
    classifier_confidence: Optional[float] = None
    geo_scores: Optional[str] = None
    classifier_scores: Optional[str] = None


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
    fields except ``classifier`` are optional so tests can supply only what
    they need.

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
        video_frame_source: Optional callable that, given the camera ``request``,
            returns the frame to record (the full-FOV, uncropped RGB frame from
            the raw stream). When ``None`` — or when it returns ``None`` — the
            recorder falls back to the cropped ``main`` frame. Injected by the
            detector so ``ml/`` stays free of camera/raw-format knowledge.
        save_confidence_threshold: Minimum classification confidence before a
            detection is saved/persisted (mutated live by the Settings page).
        ignore_species: Species (lower-cased) that are never saved even when
            classified (mutated live by the Settings page).
        geo_adjuster: Optional geomodel Bayesian-update adjuster. When set, the
            classifier's distribution is reweighted by the location's occurrence
            prior and the posterior argmax becomes the prediction; when ``None``
            (no location configured / prior unavailable) classification is
            unchanged.
    """

    classifier: Classifier
    tracker: StableDetectionTracker = field(default=None)  # type: ignore[assignment]
    classify_fn: ClassifyFn = field(default=None)  # type: ignore[assignment]
    detection_writer: Optional["DetectionWriter"] = None
    best_frame_selector: Optional[BestFrameSelector] = None
    record_fn: Optional[Callable[[str], bool]] = None
    video_frame_fn: Optional[Callable[[np.ndarray], None]] = None
    video_frame_source: Optional[Callable[[Any], Optional[np.ndarray]]] = None
    save_confidence_threshold: float = DEFAULT_SAVE_CONFIDENCE_THRESHOLD
    ignore_species: set[str] = field(default_factory=set)
    geo_adjuster: Optional["GeoPriorAdjuster"] = None

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
    result = _predict_species(context, roi)
    if track is not None:
        context.tracker.mark_classified(track.track_id, species=result.species)
    return result


def _classifier_probs(classifier: Classifier, roi: np.ndarray) -> Optional[np.ndarray]:
    """Return the classifier's full softmax vector, or ``None`` when unsupported.

    Real :class:`Classifier` instances expose ``predict_proba`` and an
    ``idx_to_class`` map; the lightweight pipeline test fakes (a bare ``object`` with
    an injected ``classify_fn``) do not. ``None`` signals the caller to fall back to
    ``classify_fn`` for the top-1-only prediction.

    Args:
        classifier: The (possibly faked) classifier from the pipeline context.
        roi: The preprocessed ROI to classify.

    Returns:
        The ``(num_classes,)`` softmax vector, or ``None`` when the classifier cannot
        produce a full distribution.
    """
    if not hasattr(classifier, "predict_proba"):
        return None
    if getattr(classifier, "idx_to_class", None) is None:
        return None
    return classifier.predict_proba(roi)


def _predict_species(context: PipelineContext, roi: np.ndarray) -> Classification:
    """Classify an ROI, applying the geomodel Bayesian update when configured.

    When the classifier can produce a full distribution the softmax is computed once
    and reused: its top-k pairs are persisted as ``classifier_scores`` and its argmax
    is the top-1 prediction. With a ``geo_adjuster`` that same distribution is then
    reweighted by the location's occurrence prior for the current week — the
    posterior argmax becomes the prediction, and the classifier's original pick plus
    the top pre-normalised scores are retained for debugging. For a test fake without
    ``predict_proba`` this falls back to the injectable ``classify_fn`` (top-1 only,
    no ``classifier_scores``).

    Args:
        context: Pipeline dependencies (classifier + classify callable + adjuster).
        roi: The preprocessed ROI to classify.

    Returns:
        The :class:`Classification` (with geomodel fields populated iff adjusted, and
        ``classifier_scores`` populated whenever a full distribution was available).
    """
    probs = _classifier_probs(context.classifier, roi)
    if probs is None:
        species, confidence = context.classify_fn(context.classifier, roi)
        return Classification(species, confidence)

    idx_to_class = context.classifier.idx_to_class
    assert idx_to_class is not None  # guaranteed by _classifier_probs
    classifier_scores = json.dumps(
        top_k_predictions(probs, idx_to_class, CLASSIFIER_TOP_K)
    )

    if context.geo_adjuster is not None:
        adjustment = context.geo_adjuster.adjust(probs, week_of_year(datetime.now()))
        return Classification(
            species=adjustment.species,
            confidence=adjustment.confidence,
            classifier_species=adjustment.classifier_species,
            classifier_confidence=adjustment.classifier_confidence,
            geo_scores=json.dumps(adjustment.top_scores),
            classifier_scores=classifier_scores,
        )

    top_idx = int(np.argmax(probs))
    return Classification(
        species=idx_to_class.get(top_idx, f"<unknown:{top_idx}>"),
        confidence=float(probs[top_idx]),
        classifier_scores=classifier_scores,
    )


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
) -> tuple[Optional[str], Optional[str]]:
    """Start a video clip for this detection, returning its path and no-clip reason.

    Args:
        context: Pipeline dependencies (holds the record callable).
        species_dir: Directory for this species' clips.
        stem: Filename stem shared by the still and clip.
        species: Species name (the relative path's directory).

    Returns:
        A ``(video_path, no_video_reason)`` pair. When recording began,
        ``video_path`` is the clip path relative to ``IMAGE_DIR`` and the reason
        is ``None``. Otherwise ``video_path`` is ``None`` and the reason is
        ``NO_VIDEO_DISABLED`` (no recorder wired) or ``NO_VIDEO_RECORDER_BUSY``
        (the single-flight recorder was already capturing another clip).
    """
    if context.record_fn is None:
        return None, NO_VIDEO_DISABLED
    started = context.record_fn(str(species_dir / f"{stem}.mp4"))
    if started:
        return f"{species}/{stem}.mp4", None
    return None, NO_VIDEO_RECORDER_BUSY


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
    video_rel, no_video_reason = _start_clip(context, species_dir, stem, result.species)

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
            no_video_reason=no_video_reason,
            track_id=track.track_id if track is not None else None,
            stable_frames=track.stable_frames if track is not None else None,
            box_x=norm[0],
            box_y=norm[1],
            box_w=norm[2],
            box_h=norm[3],
            classifier_species=result.classifier_species,
            classifier_confidence=result.classifier_confidence,
            geo_scores=result.geo_scores,
            classifier_scores=result.classifier_scores,
        )
    )


def process_single_detection_with_stable_tracks(
    item: tuple,
    context: PipelineContext,
) -> None:
    """Process a detection using multi-frame stable-track gating.

    Gates classification until the track has been stable for the tracker's
    configured number of frames, classifies the track's best observed frame, and
    persists the result when it clears the confidence threshold.

    Args:
        item: Tuple of (image, detection_id, detection, labels, classifier_class).
        context: Injected pipeline dependencies.
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

    if is_bird and result is not None and _should_persist(result, context):
        _persist_detection(context, still, detection, track, result)


def _should_persist(result: Classification, context: PipelineContext) -> bool:
    """Return whether a classification clears the save threshold and ignore list.

    Args:
        result: The classification to weigh.
        context: Pipeline dependencies (holds the live save threshold + ignore
            list).

    Returns:
        ``True`` when there is a confident, non-ignored species to persist.
    """
    if not result.species or not result.confidence:
        return False
    if result.confidence <= context.save_confidence_threshold:
        return False
    return result.species.lower() not in context.ignore_species


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
        # while idle) so a triggered clip has pre-roll footage. The clip records
        # the full, uncropped field of view (from the raw stream via
        # ``video_frame_source``) rather than the cropped ``main`` frame; if no
        # source is wired or it fails, fall back to the cropped frame.
        if context.video_frame_fn is not None:
            clip_frame = full_img
            if context.video_frame_source is not None:
                produced = context.video_frame_source(request)
                if produced is not None:
                    clip_frame = produced
            context.video_frame_fn(clip_frame)

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
    ) -> None:
        """Initialize the ClassificationManager.

        Args:
            context: Injected pipeline dependencies (see :class:`PipelineContext`).
            use_multithreading: If True, process detections on a background thread.
            queue_maxsize: Maximum queue size for async processing (0 = unlimited).
        """
        self.context = context
        self.use_multithreading = use_multithreading
        self._queue: "Queue[tuple] | None" = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

        if self.use_multithreading:
            from queue import Queue

            self._stop_event = threading.Event()
            self._queue = Queue(maxsize=queue_maxsize)
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

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
            process_single_detection_with_stable_tracks(item, context=self.context)
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
