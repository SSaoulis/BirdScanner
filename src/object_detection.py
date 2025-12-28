"""Object detection and bird classification module."""

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from classification import Classifier, ONNXClassifier, build_preprocessing


# Global state
last_detections = []
classification_results = {}
last_detection_classifications = []  # List of (box, species, confidence) tuples for temporal filtering


# ---- New multi-frame stability tracking (kept alongside the old temporal cache) ----


@dataclass
class StableTrack:
    """A lightweight track for a single object across frames."""

    track_id: int
    box: tuple
    stable_frames: int = 1
    classified: bool = False
    frames_since_seen: int = 0
    species: Optional[str] = None


def match_detection_to_track(
    detection_box: tuple,
    tracks: list[StableTrack],
    *,
    iou_threshold: float,
) -> Optional[StableTrack]:
    """Return the best matching track for a detection, or None if no match."""

    best_track: Optional[StableTrack] = None
    best_iou = 0.0
    for t in tracks:
        score = iou(detection_box, t.box)
        if score > best_iou:
            best_iou = score
            best_track = t
    if best_track is None or best_iou <= iou_threshold:
        return None
    return best_track


def update_tracks_for_frame(
    detection_boxes: list[tuple],
    tracks: list[StableTrack],
    *,
    iou_threshold: float,
    max_missing_frames: int = 0,
    on_track_deleted: Optional[Callable[[StableTrack], None]] = None,
) -> tuple[list[StableTrack], dict[int, StableTrack]]:
    """Update track state from a list of detection boxes for a single frame.

    Args:
        detection_boxes: List of detection boxes for the frame.
        tracks: Existing tracks to update.
        iou_threshold: IoU threshold to consider a match.
        max_missing_frames: If > 0, delete tracks that have not been matched for this many frames.
        on_track_deleted: Optional callback invoked for each deleted track.

    Returns:
        (updated_tracks, per_detection_track_map)
    """

    per_det_track: dict[int, StableTrack] = {}
    used_track_ids: set[int] = set()

    # Age all tracks; any track that is matched this frame will be reset to 0.
    for t in tracks:
        t.frames_since_seen += 1

    next_track_id = (max((t.track_id for t in tracks), default=-1) + 1) if tracks else 0

    for det_id, box in enumerate(detection_boxes):
        match = match_detection_to_track(box, tracks, iou_threshold=iou_threshold)
        if match is None or match.track_id in used_track_ids:
            new_track = StableTrack(
                track_id=next_track_id,
                box=box,
                stable_frames=1,
                classified=False,
                frames_since_seen=0,
                species=None,
            )
            next_track_id += 1
            tracks.append(new_track)
            per_det_track[det_id] = new_track
            used_track_ids.add(new_track.track_id)
        else:
            match.box = box
            match.stable_frames += 1
            match.frames_since_seen = 0
            per_det_track[det_id] = match
            used_track_ids.add(match.track_id)

    # Drop tracks that haven't been seen recently.
    if max_missing_frames and max_missing_frames > 0:
        kept: list[StableTrack] = []
        for t in tracks:
            if t.frames_since_seen <= max_missing_frames:
                kept.append(t)
            else:
                if on_track_deleted is not None:
                    on_track_deleted(t)
        tracks = kept

    return tracks, per_det_track


def should_classify_track(track: StableTrack, *, min_stable_frames: int) -> bool:
    """Whether a track has been stable long enough and not yet classified."""

    return (not track.classified) and track.stable_frames >= min_stable_frames


class StableDetectionTracker:
    """Tracks detections across frames and gates classification until stable."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.6,
        min_stable_frames: int = 3,
        max_missing_frames: int = 0,
        on_track_became_stable: Optional[Callable[[StableTrack], None]] = None,
        on_track_deleted: Optional[Callable[[StableTrack], None]] = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.min_stable_frames = min_stable_frames
        self.max_missing_frames = max_missing_frames
        self.on_track_became_stable = on_track_became_stable
        self.on_track_deleted = on_track_deleted
        self._tracks: list[StableTrack] = []
        self._last_per_det_track: dict[int, StableTrack] = {}

    def update_frame(self, detections: list) -> None:
        """Update tracker state with a list of Detection-like objects (must have .box)."""

        boxes = [d.box for d in detections]

        # Snapshot which tracks were already stable so we can detect first-time stability.
        previously_stable_ids = {
            t.track_id for t in self._tracks if t.stable_frames >= self.min_stable_frames
        }

        self._tracks, self._last_per_det_track = update_tracks_for_frame(
            boxes,
            self._tracks,
            iou_threshold=self.iou_threshold,
            max_missing_frames=self.max_missing_frames,
            on_track_deleted=self.on_track_deleted,
        )

        # Fire 'became stable' event once per track, at the moment it crosses the threshold.
        if self.on_track_became_stable is not None:
            for t in self._tracks:
                if (
                    t.track_id not in previously_stable_ids
                    and t.stable_frames >= self.min_stable_frames
                ):
                    self.on_track_became_stable(t)

    def track_for_detection_id(self, detection_id: int) -> Optional[StableTrack]:
        return self._last_per_det_track.get(detection_id)

    def mark_classified(self, track_id: int, *, species: Optional[str] = None) -> None:
        for t in self._tracks:
            if t.track_id == track_id:
                t.classified = True
                if species is not None:
                    t.species = species
                return

    def track_count(self) -> int:
        """Test helper: the number of currently active tracks."""

        return len(self._tracks)


# Global tracker instance used by default in the live pipeline.
# Tests can instantiate StableDetectionTracker directly.
stable_detection_tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=3)


def should_run_bird_classification_for_detection(
    detection_id: int,
    *,
    tracker: StableDetectionTracker,
) -> bool:
    """New gating logic: only classify if the detection belongs to a stable track."""

    track = tracker.track_for_detection_id(detection_id)
    if track is None:
        return False
    return should_classify_track(track, min_stable_frames=tracker.min_stable_frames)


def process_single_detection_with_stable_tracks(
    item: tuple,
    *,
    results_lock: threading.Lock,
    classifier: Classifier,
    tracker: StableDetectionTracker,
    classify_fn: Optional[Callable[[Classifier, np.ndarray], tuple]] = None,
) -> None:
    """Process detection using the new multi-frame stable-track gating logic.

    The existing, older per-frame cache logic is intentionally left in
    `process_single_detection` for reference.
    """

    if classify_fn is None:
        classify_fn = run_bird_classification

    image, detection_id, detection, labels, classifier_class = item

    species = None
    confidence = None

    # Gate classification until stable over N frames.
    if classifier_class.lower() == "bird" and should_run_bird_classification_for_detection(
        detection_id, tracker=tracker
    ):
        roi, coords = preprocess_roi(image, detection.box)
        species, confidence = classify_fn(classifier, roi)

        # Ensure we only classify a stable track once.
        track = tracker.track_for_detection_id(detection_id)
        if track is not None:
            tracker.mark_classified(track.track_id, species=species)

    # Always draw boxes; include optional classification result.
    roi, coords = preprocess_roi(image, detection.box)
    image_with_boxes = draw_boxes(image.copy(), coords, detection, labels, species, confidence)

    # Save only after a classification actually happened.
    if classifier_class.lower() == "bird" and species and confidence and confidence > 0.6:
        time = datetime.now()
        os.makedirs(f"/home/stefan/Pictures/bird_detections/{species}/", exist_ok=True)
        output_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"/home/stefan/Pictures/bird_detections/{species}/{time}.png", output_image)

    with results_lock:
        classification_results[detection_id] = (species, confidence)


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


def iou(box1: tuple, box2: tuple) -> float:
    """Calculate Intersection over Union (IoU) between two boxes.
    
    Computes the IoU metric for two bounding boxes, commonly used for
    overlap detection and temporal filtering.
    
    Args:
        box1: Bounding box in format (x, y, w, h).
        box2: Bounding box in format (x, y, w, h).
        
    Returns:
        float: IoU score between 0.0 and 1.0, where 1.0 indicates complete overlap.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2) format
    box1_x1, box1_y1, box1_x2, box1_y2 = x1, y1, x1 + w1, y1 + h1
    box2_x1, box2_y1, box2_x2, box2_y2 = x2, y2, x2 + w2, y2 + h2
    
    # Calculate intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


class Detection:
    """Represents a detected object with bounding box and category."""
    
    def __init__(self, coords: np.ndarray, category: int, conf: float, metadata: dict) -> None:
        """Create a Detection object, recording the bounding box, category and confidence.
        
        Args:
            coords: Normalized coordinates from inference as numpy array.
            category: Object category index from model output.
            conf: Confidence score from model (0.0 to 1.0).
            metadata: Camera metadata for coordinate conversion.
        """
        self.category = category
        self.conf = conf
        # Store IMX500 and metadata for later coordinate conversion
        self._coords = coords
        self._metadata = metadata
        self.box = None  # Will be set by set_box() after IMX500 is available

    def set_box(self, box: tuple) -> None:
        """Set the box after coordinate conversion.
        
        Args:
            box: Converted bounding box in format (x, y, w, h).
        """
        self.box = box


def parse_detections(
    metadata: dict,
    imx500,
    intrinsics,
    threshold: float,
    picam2,
) -> list:
    """Parse the output tensor into detected objects scaled to ISP output.
    
    Extracts bounding boxes, confidence scores, and class indices from the
    IMX500 inference output, filters by confidence threshold, and converts
    normalized coordinates to ISP output coordinates.
    
    Args:
        metadata: Camera metadata containing inference output tensors.
        imx500: IMX500 device instance for coordinate conversion.
        intrinsics: Network intrinsics with bbox normalization and order settings.
        threshold: Confidence threshold for filtering detections.
        picam2: Picamera2 instance for coordinate system reference.
        
    Returns:
        list: List of Detection objects with converted coordinates.
    """
    global last_detections
    
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    
    if np_outputs is None:
        return last_detections
    
    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
    
    if bbox_normalization:
        boxes = boxes / input_h
    
    if bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]
    
    boxes = np.array_split(boxes, 4, axis=1)
    boxes = zip(*boxes)
    
    last_detections = []
    for box, score, category in zip(boxes, scores, classes):
        if score > threshold:
            detection = Detection(box, category, score, metadata) # type: ignore
            detection.set_box(imx500.convert_inference_coords(box, metadata, picam2))
            last_detections.append(detection)
    
    return last_detections

def get_labels(intrinsics) -> list:
    """Get labels from intrinsics, filtering empty ones.
    
    Extracts and filters the class labels from network intrinsics,
    removing empty strings and placeholder '-' labels.
    
    Args:
        intrinsics: Network intrinsics containing label information.
        
    Returns:
        list: Filtered list of class label strings.
    """
    labels = intrinsics.labels
    labels = [label for label in labels if label and label != "-"]
    return labels


def preprocess_roi(image: np.ndarray, box: tuple) -> tuple:
    """Preprocess the region of interest for classification.
    
    Extracts and preprocesses a detection region by:
    1. Converting to square by using the larger dimension
    2. Expanding by 20% (10% on each side) through the center
    3. Clamping to image boundaries while maintaining square shape
    
    Args:
        image: Input image as numpy array.
        box: Bounding box in format (x, y, w, h).
        
    Returns:
        tuple: (roi, coords) where roi is the preprocessed image patch
            and coords is the final box in format (x, y, size, size).
    """
    x, y, w, h = box
    img_h, img_w = image.shape[:2]

    # Step 1: Make it square by using the maximum side
    max_side = max(w, h)

    # Center the smaller dimension
    x_offset = (max_side - w) // 2
    y_offset = (max_side - h) // 2

    x1 = x - x_offset
    y1 = y - y_offset
    x2 = x1 + max_side
    y2 = y1 + max_side
    
    # Step 2: Expand by 20% from center
    expansion = int(max_side * 0.2 / 2)  # 10% on each side
    
    x1 -= expansion
    y1 -= expansion
    x2 += expansion
    y2 += expansion

    expanded_size = x2 - x1

    # Step 3: Clamp to image boundaries while keeping it square
    if x1 < 0:
        x1 = 0
        x2 = expanded_size
    if y1 < 0:
        y1 = 0
        y2 = expanded_size
    if x2 > img_w:
        x2 = img_w
        x1 = max(0, x2 - expanded_size)
    if y2 > img_h:
        y2 = img_h
        y1 = max(0, y2 - expanded_size)

    # Ensure square: take the smaller of the two clamped dimensions
    final_size = min(x2 - x1, y2 - y1)

    # Re-center if we had to clamp
    if final_size < expanded_size:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        x1 = int(cx - final_size / 2)
        y1 = int(cy - final_size / 2)
        x2 = x1 + final_size
        y2 = y1 + final_size

        # Final clamp to image boundaries
        x1 = max(0, min(x1, img_w - final_size))
        y1 = max(0, min(y1, img_h - final_size))
        x2 = min(x1 + final_size, img_w)
        y2 = min(y1 + final_size, img_h)

    roi = image[y1:y2, x1:x2]
    return roi, (x1, y1, final_size, final_size)


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


def draw_boxes(
    image_array: np.ndarray,
    coords: tuple,
    detection: Detection,
    labels: list,
    species: Optional[str] = None,
    confidence: Optional[float] = None,
) -> np.ndarray:
    """Draw detection boxes and labels on image array.
    
    Draws bounding boxes, class labels, and optional classification results
    on the image. Includes a semi-transparent background for text readability.
    
    Args:
        image_array: Input image as numpy array (modified in-place).
        coords: Box coordinates in format (x, y, w, h).
        detection: Detection object with category and confidence.
        labels: List of class label strings.
        species: Optional species name from classification.
        confidence: Optional classification confidence score.
        
    Returns:
        np.ndarray: The modified image array with drawn boxes and labels.
    """
    x, y, w, h = coords
    overlay = image_array.copy()
    label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
    if confidence is not None:
        label += f" - {confidence:.2f}"
    if species:
        label += f" - {species}"

    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_x = x + 5
    text_y = y + 15

    cv2.rectangle(
        overlay,
        (text_x, text_y - text_height),
        (text_x + text_width, text_y + baseline),
        (255, 255, 255),
        cv2.FILLED,
    )
    alpha = 0.30
    cv2.addWeighted(overlay, alpha, image_array, 1 - alpha, 0, image_array)

    cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)
    cv2.putText(image_array, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return image_array


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
    global last_detection_classifications
    image, detection_id, detection, labels, classifier_class = item

    # Temporal filtering: reuse classification if box overlaps significantly with any detection from last frame
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
    image_with_boxes = draw_boxes(image.copy(), coords, detection, labels, species, confidence)

    if classifier_class.lower() == "bird" and species and confidence:
        if confidence > 0.6:
            time = datetime.now()
            os.makedirs(f"/home/stefan/Pictures/bird_detections/{species}/", exist_ok=True)
            output_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"/home/stefan/Pictures/bird_detections/{species}/{time}.png", output_image)
    
    with results_lock:
        classification_results[detection_id] = (species, confidence)


def update_detection_classifications_cache(
    detections: list,
    classification_results: dict,
) -> None:
    """Update the cache of detection classifications for the current frame.
    
    Builds a list of (box, species, confidence) tuples from the current
    detections and their classification results, replacing the previous
    frame's cache.
    
    Args:
        detections: List of Detection objects from current frame.
        classification_results: Dictionary mapping detection_id to (species, confidence).
    """
    global last_detection_classifications
    
    new_classifications = []
    for detection_id, detection in enumerate(detections):
        if detection_id in classification_results:
            species, confidence = classification_results[detection_id]
            if species and confidence:
                new_classifications.append((detection.box, species, confidence))
    
    last_detection_classifications = new_classifications


def process_detections(
    request,
    stream: str,
    last_results: list,
    manager: 'ClassificationManager',
    labels: list,
    full_img_processor=None,
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
        full_img_processor: Optional callback to process full image before detection processing.
    """
    if last_results is None:
        return
    from picamera2 import MappedArray  # type: ignore

    with MappedArray(request, stream) as m:
        full_img = m.array.copy()
        
        for detection_id, detection in enumerate(last_results):
            _, coords = preprocess_roi(full_img, detection.box)
            image_with_boxes = draw_boxes(full_img.copy(), coords, detection, labels)
            m.array[:] = image_with_boxes

            classifier_class = labels[int(detection.category)]
            if classifier_class.lower() == "bird":
                manager.process((full_img, detection_id, detection, labels, classifier_class))


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
        """
        self.classifier = classifier
        self.use_multithreading = use_multithreading
        self.use_stable_track_gating = use_stable_track_gating
        self.tracker = tracker or stable_detection_tracker
        self._results_lock = None
        self._queue = None
        self._thread = None
        self._stop_event = None

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
            if self.use_stable_track_gating:
                process_single_detection_with_stable_tracks(
                    item,
                    results_lock=self._results_lock,  # type: ignore[arg-type]
                    classifier=self.classifier,
                    tracker=self.tracker,
                )
            else:
                process_single_detection(item, results_lock=self._results_lock, classifier=self.classifier)  # type: ignore
            return

        from queue import Full

        try:
            self._queue.put_nowait(item)  # type: ignore
        except Full:
            return

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

            if self.use_stable_track_gating:
                process_single_detection_with_stable_tracks(
                    item,
                    results_lock=self._results_lock,  # type: ignore[arg-type]
                    classifier=self.classifier,
                    tracker=self.tracker,
                )
            else:
                process_single_detection(item, results_lock=self._results_lock, classifier=self.classifier)  # type: ignore

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
