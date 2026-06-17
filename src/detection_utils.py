"""Shared geometry and drawing helpers for the detection pipeline.

These utilities operate on bounding boxes in ``(x, y, w, h)`` ISP-output pixel
coordinates and on image arrays. They are intentionally free of any tracking or
classification state so they can be reused across the pipeline and unit-tested
in isolation.
"""

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from object_detection import Detection


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


def draw_boxes(
    image_array: np.ndarray,
    coords: tuple,
    detection: "Detection",
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

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
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
    cv2.putText(
        image_array,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
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
