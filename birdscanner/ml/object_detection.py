"""Core object detection: parsing IMX500 inference output into Detection objects.

This module is deliberately narrow — it owns the ``Detection`` data model and the
parsing of the IMX500 inference tensor. Supporting concerns live elsewhere:

- ``detection_utils`` — IoU, ROI preprocessing, box drawing, thumbnails
- ``tracking`` — multi-frame stable-track gating (``StableDetectionTracker``)
- ``classification_pipeline`` — species classification dispatch and persistence
"""

from typing import Iterable

import numpy as np

from birdscanner.ml.detection_utils import label_for_category

# Most recent set of parsed detections; returned as a fallback when a frame
# yields no inference output so the live loop always has something to draw.
last_detections: list["Detection"] = []


class Detection:
    """Represents a detected object with bounding box and category."""

    def __init__(
        self, coords: np.ndarray, category: int, conf: float, metadata: dict
    ) -> None:
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
        self.box: tuple | None = None  # Set by set_box() after IMX500 is available

    def set_box(self, box: tuple) -> None:
        """Set the box after coordinate conversion.

        Args:
            box: Converted bounding box in format (x, y, w, h).
        """
        self.box = box


def _decode_boxes(boxes: np.ndarray, intrinsics, input_h: int):
    """Normalise and re-order the raw box tensor into per-detection tuples.

    Args:
        boxes: The raw ``(N, 4)`` box tensor from the inference output.
        intrinsics: Network intrinsics with bbox normalization and order settings.
        input_h: The network input height (for coordinate normalization).

    Returns:
        An iterator of per-detection ``(y0, x0, y1, x1)`` box tuples.
    """
    if intrinsics.bbox_normalization:
        boxes = boxes / input_h
    if intrinsics.bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]
    return zip(*np.array_split(boxes, 4, axis=1))


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

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections

    _, input_h = imx500.get_input_size()
    boxes = _decode_boxes(np_outputs[0][0], intrinsics, input_h)
    scores, classes = np_outputs[1][0], np_outputs[2][0]

    last_detections = []
    for box, score, category in zip(boxes, scores, classes):
        if score > threshold:
            detection = Detection(box, category, score, metadata)  # type: ignore
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


def filter_included_detections(
    detections: list,
    labels: list,
    included: Iterable[str],
) -> list:
    """Keep only detections whose object-detection class is in the include list.

    The IMX500 YOLO model emits every COCO object class it sees (bird, person,
    car, bench, ...), not just birds. Anything that is not an included class
    otherwise enters the tracker — creating tracks that spam the ``tracking``
    logs with stable/deleted events and draw boxes on the preview. Filtering
    here, at the single point the parsed detections re-enter the pipeline, keeps
    unwanted classes out of the tracker, the logs, the drawing, and
    classification entirely (an allowlist, e.g. just ``"bird"``).

    Matching is case-insensitive against the (filtered) label list. A detection
    whose category index is out of range for ``labels`` is kept (it is handled
    later by the out-of-range guard in the classification pipeline).

    Args:
        detections: Detection objects for the current frame.
        labels: Class label strings (index-aligned with ``detection.category``).
        included: Class labels to keep (compared lower-cased). Empty means keep
            everything (a no-op), so clearing the list never silently drops all
            detections.

    Returns:
        The detections whose class is included (the same list object when the
        include set is empty, so the no-op case allocates nothing).
    """
    included_lower = {name.lower() for name in included}
    if not included_lower:
        return detections

    kept = []
    for detection in detections:
        label = label_for_category(labels, int(detection.category))
        if label is not None and label.lower() not in included_lower:
            continue
        kept.append(detection)
    return kept
