"""Object detection and bird classification module."""

import os
import threading
from datetime import datetime

import cv2
import numpy as np
from picamera2 import MappedArray

from classification import Classifier, ONNXClassifier, build_preprocessing


# Global state
last_detections = []
classification_results = {}
last_bird_classification = None  # (box, species, confidence) for temporal filtering


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
    preprocessing = build_preprocessing({
        "size": (384, 384),
        "rgb_values": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "center_crop": 1.0,
        "simple_crop": False,
    })
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
            detection = Detection(box, category, score, metadata)
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
    species: str = None,
    confidence: float = None,
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
    
    Args:
        item: Tuple of (image, detection_id, detection, labels, classifier_class).
        results_lock: Thread lock for safe results dictionary access.
        classifier: Classifier instance for bird species classification.
    """
    global last_bird_classification
    image, detection_id, detection, labels, classifier_class = item

    # Temporal filtering: reuse classification if box overlaps significantly with last detection
    species = None
    confidence = None
    
    if last_bird_classification is not None:
        last_box, last_species, last_confidence = last_bird_classification # type: ignore
        if iou(detection.box, last_box) > 0.8:
            # Reuse classification from previous frame
            species = last_species
            confidence = last_confidence
    
    # Run classification only if we didn't reuse
    if species is None:
        roi, coords = preprocess_roi(image, detection.box)
        species, confidence = run_bird_classification(classifier, roi)
        # Update last classification for next frame
        with results_lock:
            last_bird_classification = (detection.box, species, confidence)
    
    roi, coords = preprocess_roi(image, detection.box)
    image_with_boxes = draw_boxes(image.copy(), coords, detection, labels, species, confidence)

    if classifier_class.lower() == "bird" and species and confidence:
        if confidence > 0.4:
            time = datetime.now()
            os.makedirs(f"/home/stefan/Pictures/bird_detections/{species}/", exist_ok=True)
            output_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"/home/stefan/Pictures/bird_detections/{species}/{time}.png", output_image)
    
    with results_lock:
        classification_results[detection_id] = (species, confidence)


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
    ) -> None:
        """Initialize the ClassificationManager.
        
        Creates a classification processor that can operate in synchronous or
        asynchronous mode. In async mode, detections are queued for processing
        by a background worker thread.
        
        Args:
            classifier: Classifier instance for bird species classification.
            use_multithreading: If True, enable async processing with background thread.
            queue_maxsize: Maximum queue size for async processing. 0 means unlimited.
        """
        self.classifier = classifier
        self.use_multithreading = use_multithreading
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
            process_single_detection(item, results_lock=self._results_lock, classifier=self.classifier)
            return

        from queue import Full
        try:
            self._queue.put_nowait(item)
        except Full:
            # Drop frame if queue is full.
            return

    def _worker_loop(self) -> None:
        """Worker thread main loop for processing queued detections.
        
        Continuously retrieves items from the queue and processes them
        until a None sentinel value is received, indicating shutdown.
        """
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            process_single_detection(item, results_lock=self._results_lock, classifier=self.classifier)
            self._queue.task_done()

    def stop(self) -> None:
        """Stop the worker thread gracefully.
        
        Signals the worker thread to stop and waits for it to finish
        with a 5-second timeout. In synchronous mode, this is a no-op.
        """
        if not self.use_multithreading:
            return
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=5)
