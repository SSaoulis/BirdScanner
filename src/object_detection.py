"""Core object detection: parsing IMX500 inference output into Detection objects.

This module is deliberately narrow — it owns the ``Detection`` data model and the
parsing of the IMX500 inference tensor. Supporting concerns live elsewhere:

- ``detection_utils`` — IoU, ROI preprocessing, box drawing, thumbnails
- ``tracking`` — multi-frame stable-track gating (``StableDetectionTracker``)
- ``classification_pipeline`` — species classification dispatch and persistence
"""

import numpy as np

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
