"""Core object detection: parsing IMX500 inference output into Detection objects.

This module is deliberately narrow — it owns the ``Detection`` data model and the
parsing of the IMX500 inference tensor. Supporting concerns live elsewhere:

- ``detection_utils`` — IoU, ROI preprocessing, box drawing, thumbnails
- ``tracking`` — multi-frame stable-track gating (``StableDetectionTracker``)
- ``classification_pipeline`` — species classification dispatch and persistence
"""

from typing import Any, Iterable, NamedTuple, Optional

import numpy as np

from birdscanner.ml.detection_utils import label_for_category

# Most recent set of parsed detections; returned as a fallback when a frame
# yields no inference output so the live loop always has something to draw.
last_detections: list["Detection"] = []


class InferenceRoi(NamedTuple):
    """The on-chip DNN inference ROI plus the sensor it is expressed in.

    When the IMX500's inference ROI is restricted (via
    ``IMX500.set_inference_roi_abs``) the network only sees the ROI sub-region,
    so its output boxes come back normalized ``[0, 1]`` **relative to the ROI**,
    not the full sensor. :func:`parse_detections` uses this to remap those boxes
    back into full-sensor fractions before ``convert_inference_coords`` (which
    assumes full-sensor normalization). The sensor dimensions travel with the ROI
    so ``ml/`` needs no import from ``detector/`` (the one-way layering holds).

    Attributes:
        left: ROI left edge, in full-sensor pixels.
        top: ROI top edge, in full-sensor pixels.
        width: ROI width, in full-sensor pixels.
        height: ROI height, in full-sensor pixels.
        sensor_w: Full sensor active-area width, in pixels.
        sensor_h: Full sensor active-area height, in pixels.
    """

    left: int
    top: int
    width: int
    height: int
    sensor_w: int
    sensor_h: int


class InferenceContext(NamedTuple):
    """The constant IMX500 handles :func:`parse_detections` reads each frame.

    Bundled into one value object so the per-frame call stays narrow: the device,
    its (overridden) network intrinsics, and the Picamera2 stream do not change
    across the capture loop, so they are built once and passed together.

    Attributes:
        imx500: The IMX500 device (output tensor access + coordinate conversion).
        intrinsics: The network intrinsics (bbox normalization + order).
        picam2: The Picamera2 instance (coordinate-system reference).
    """

    imx500: Any
    intrinsics: Any
    picam2: Any


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


def _remap_roi_relative_box(box, roi: InferenceRoi) -> np.ndarray:
    """Map a box normalized to the inference ROI back to full-sensor fractions.

    The on-chip network, when its input is restricted to ``roi``, returns box
    coordinates as fractions of the ROI. ``convert_inference_coords`` instead
    expects fractions of the full sensor, so each coordinate is placed inside the
    ROI (``roi.left + frac * roi.width``) and renormalized by the sensor size.
    This is the identity when ``roi`` spans the whole sensor.

    Args:
        box: The decoded ``(y0, x0, y1, x1)`` box in ``[0, 1]`` ROI fractions
            (the tuple order ``convert_inference_coords`` unpacks).
        roi: The active inference ROI and the sensor it is expressed in.

    Returns:
        The ``(y0, x0, y1, x1)`` box as full-sensor ``[0, 1]`` fractions.
    """
    y0, x0, y1, x1 = (float(v) for v in box)
    return np.array(
        [
            (roi.top + y0 * roi.height) / roi.sensor_h,
            (roi.left + x0 * roi.width) / roi.sensor_w,
            (roi.top + y1 * roi.height) / roi.sensor_h,
            (roi.left + x1 * roi.width) / roi.sensor_w,
        ]
    )


def parse_detections(
    metadata: dict,
    context: InferenceContext,
    threshold: float,
    inference_roi: Optional[InferenceRoi] = None,
) -> list:
    """Parse the output tensor into detected objects scaled to ISP output.

    Extracts bounding boxes, confidence scores, and class indices from the
    IMX500 inference output, filters by confidence threshold, and converts
    normalized coordinates to ISP output coordinates.

    When ``inference_roi`` is supplied, the on-chip network only saw that ROI
    sub-region, so its boxes are normalized to the ROI; each is remapped back to
    full-sensor fractions (:func:`_remap_roi_relative_box`) before
    ``convert_inference_coords`` — which assumes full-sensor normalization — so
    the box lands on the bird rather than being mis-scaled and clamped to the ROI
    edge. ``None`` (no ROI restriction) leaves the coordinates untouched.

    Args:
        metadata: Camera metadata containing inference output tensors.
        context: The IMX500 device, intrinsics, and Picamera2 stream (constant
            across the capture loop).
        threshold: Confidence threshold for filtering detections.
        inference_roi: The active on-chip inference ROI (and its sensor), when the
            DNN input is restricted to a crop; ``None`` when it sees the full FOV.

    Returns:
        list: List of Detection objects with converted coordinates.
    """
    global last_detections
    imx500 = context.imx500

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections

    _, input_h = imx500.get_input_size()
    boxes = _decode_boxes(np_outputs[0][0], context.intrinsics, input_h)
    scores, classes = np_outputs[1][0], np_outputs[2][0]

    last_detections = []
    for box, score, category in zip(boxes, scores, classes):
        if score > threshold:
            if inference_roi is not None:
                box = _remap_roi_relative_box(box, inference_roi)
            detection = Detection(box, category, score, metadata)  # type: ignore
            detection.set_box(
                imx500.convert_inference_coords(box, metadata, context.picam2)
            )
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
