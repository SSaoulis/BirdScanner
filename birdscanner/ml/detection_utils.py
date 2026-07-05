"""Shared geometry and drawing helpers for the detection pipeline.

Bounding-box maths lives on the :class:`Box` value object; the module-level
functions (``iou``, ``preprocess_roi``, ``normalized_box``, ``draw_boxes``) are
thin, tuple-based wrappers around it so existing call sites keep passing plain
``(x, y, w, h)`` tuples. Everything here is free of tracking/classification
state so it can be reused across the pipeline and unit-tested in isolation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from birdscanner.ml.object_detection import Detection

# Fraction the ROI is padded by when squared up for the classifier (10% a side).
_ROI_PAD_FRAC = 0.2


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the closed ``[low, high]`` interval."""
    return max(low, min(high, value))


@dataclass(frozen=True)
class Box:
    """An axis-aligned bounding box in ``(x, y, w, h)`` pixel coordinates.

    The box owns every geometric operation the pipeline needs (overlap, square
    padding, cropping, normalisation) so its callers stay declarative and their
    per-function local-variable count stays small.
    """

    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_xywh(cls, box: Tuple[float, float, float, float]) -> "Box":
        """Build a :class:`Box` from an ``(x, y, w, h)`` tuple."""
        x, y, w, h = box
        return cls(x, y, w, h)

    @property
    def x2(self) -> float:
        """Right edge (``x + w``)."""
        return self.x + self.w

    @property
    def y2(self) -> float:
        """Bottom edge (``y + h``)."""
        return self.y + self.h

    @property
    def area(self) -> float:
        """Box area (``w * h``)."""
        return self.w * self.h

    def iou(self, other: "Box") -> float:
        """Return the Intersection-over-Union overlap with ``other`` in ``[0, 1]``.

        Args:
            other: The box to compare against.

        Returns:
            The IoU score; ``0.0`` when the boxes are disjoint or degenerate.
        """
        inter_w = max(0.0, min(self.x2, other.x2) - max(self.x, other.x))
        inter_h = max(0.0, min(self.y2, other.y2) - max(self.y, other.y))
        intersection = inter_w * inter_h
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def padded_square(self, img_w: int, img_h: int) -> "Box":
        """Return a square, ``_ROI_PAD_FRAC``-padded box centred on this one.

        The result is expanded to a square using the longer side, padded, then
        clamped so it stays fully inside a ``img_w`` x ``img_h`` image while
        remaining square. Integer-aligned so it can index a numpy array.

        Args:
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            The padded, clamped square box (integer coordinates).
        """
        size = min(max(self.w, self.h) * (1.0 + _ROI_PAD_FRAC), img_w, img_h)
        left = _clamp(self.x + self.w / 2 - size / 2, 0.0, max(0.0, img_w - size))
        top = _clamp(self.y + self.h / 2 - size / 2, 0.0, max(0.0, img_h - size))
        return Box(int(left), int(top), int(size), int(size))

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the image patch covered by this box (a numpy view)."""
        x1, y1 = int(self.x), int(self.y)
        return image[y1 : y1 + int(self.h), x1 : x1 + int(self.w)]

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        """Return the box as an integer ``(x, y, w, h)`` tuple."""
        return (int(self.x), int(self.y), int(self.w), int(self.h))

    def normalized(
        self, image_shape: Tuple[int, ...]
    ) -> Tuple[float, float, float, float]:
        """Return the box as ``[0, 1]`` fractions of the image dimensions.

        Args:
            image_shape: The image's numpy shape, i.e. ``(height, width, ...)``.

        Returns:
            ``(x, y, w, h)`` clamped to ``[0, 1]``; all zeros for a degenerate
            image.
        """
        img_h, img_w = image_shape[:2]
        if img_w <= 0 or img_h <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            _clamp(self.x / img_w, 0.0, 1.0),
            _clamp(self.y / img_h, 0.0, 1.0),
            _clamp(self.w / img_w, 0.0, 1.0),
            _clamp(self.h / img_h, 0.0, 1.0),
        )


def label_for_category(labels: list, category: int) -> Optional[str]:
    """Return the label for a class index, or ``None`` if it is out of range.

    The IMX500 SSD model occasionally emits spurious detections with a class
    index outside the (filtered) label list. Indexing ``labels`` directly with
    such a value raises ``IndexError`` and crashes the camera callback, so all
    label look-ups go through this bounds-checked helper.

    Args:
        labels: List of class label strings.
        category: Class index from a detection.

    Returns:
        Optional[str]: The matching label, or ``None`` when the index is out of
        range for ``labels``.
    """
    if 0 <= category < len(labels):
        return labels[category]
    return None


def normalized_box(box: tuple, image_shape: tuple) -> tuple:
    """Convert a pixel-space box to fractions of the image dimensions.

    The detection box is stored normalized so the frontend can overlay it on the
    saved image at any rendered size without needing the original pixel
    dimensions. The result is clamped to ``[0, 1]``.

    Args:
        box: Bounding box in ``(x, y, w, h)`` pixel coordinates.
        image_shape: The image's ``numpy`` shape, i.e. ``(height, width, ...)``.

    Returns:
        tuple: ``(x, y, w, h)`` as fractions in ``[0, 1]`` of the image's width
        and height.
    """
    return Box.from_xywh(box).normalized(image_shape)


def iou(box1: tuple, box2: tuple) -> float:
    """Calculate Intersection over Union (IoU) between two boxes.

    Args:
        box1: Bounding box in format (x, y, w, h).
        box2: Bounding box in format (x, y, w, h).

    Returns:
        float: IoU score between 0.0 and 1.0, where 1.0 indicates complete overlap.
    """
    return Box.from_xywh(box1).iou(Box.from_xywh(box2))


def preprocess_roi(image: np.ndarray, box: tuple) -> tuple:
    """Preprocess the region of interest for classification.

    Expands the detection box to a padded square (see :meth:`Box.padded_square`)
    centred on the box, clamped to the image bounds, and crops it.

    Args:
        image: Input image as numpy array.
        box: Bounding box in format (x, y, w, h).

    Returns:
        tuple: (roi, coords) where roi is the preprocessed image patch and coords
            is the final box in format (x, y, size, size).
    """
    img_h, img_w = image.shape[:2]
    square = Box.from_xywh(box).padded_square(img_w, img_h)
    return square.crop(image), square.as_int_tuple()


def _format_label(
    detection: "Detection",
    labels: list,
    classification: Optional[Tuple[Optional[str], Optional[float]]],
) -> str:
    """Build the annotation caption for a detection box.

    Args:
        detection: Detection object exposing ``category`` and ``conf``.
        labels: List of class label strings.
        classification: Optional ``(species, confidence)`` from the classifier.

    Returns:
        The formatted label string (e.g. ``"bird (0.91) - 0.87 - Robin"``).
    """
    category = int(detection.category)
    label_name = label_for_category(labels, category) or f"id:{category}"
    label = f"{label_name} ({detection.conf:.2f})"
    species, confidence = classification or (None, None)
    if confidence is not None:
        label += f" - {confidence:.2f}"
    if species:
        label += f" - {species}"
    return label


def _draw_caption(image_array: np.ndarray, origin: Tuple[int, int], text: str) -> None:
    """Draw ``text`` at ``origin`` over a semi-transparent white plate, in place.

    Args:
        image_array: Image to annotate (modified in place).
        origin: The box's top-left ``(x, y)`` corner.
        text: The caption to render.
    """
    text_x, text_y = origin[0] + 5, origin[1] + 15
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    overlay = image_array.copy()
    cv2.rectangle(
        overlay,
        (text_x, text_y - height),
        (text_x + width, text_y + baseline),
        (255, 255, 255),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.30, image_array, 0.70, 0, image_array)
    cv2.putText(
        image_array,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )


def draw_boxes(
    image_array: np.ndarray,
    coords: tuple,
    detection: "Detection",
    labels: list,
    classification: Optional[Tuple[Optional[str], Optional[float]]] = None,
) -> np.ndarray:
    """Draw a detection box and its label on ``image_array`` in place.

    Args:
        image_array: Input image as numpy array (modified in-place).
        coords: Box coordinates in format (x, y, w, h).
        detection: Detection object with ``category`` and ``conf``.
        labels: List of class label strings.
        classification: Optional ``(species, confidence)`` classification result.

    Returns:
        np.ndarray: The modified image array with the drawn box and label.
    """
    x, y, w, h = coords
    _draw_caption(image_array, (x, y), _format_label(detection, labels, classification))
    cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)
    return image_array


def save_thumbnail(roi: np.ndarray, output_path: str) -> None:
    """Save a 200×200 JPEG thumbnail of the given ROI.

    Args:
        roi: The region-of-interest as an RGB numpy array.
        output_path: Absolute filesystem path for the output JPEG.
    """
    img = Image.fromarray(roi)
    img = img.resize((200, 200), Image.Resampling.LANCZOS)
    img.save(output_path, "JPEG", quality=85)
