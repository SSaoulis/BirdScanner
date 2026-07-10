"""Off-Pi object detector standing in for the IMX500's on-chip ``.rpk``.

The real detector runs YOLO11n on the IMX500 silicon and cannot execute off the
Pi.  For local development / testing this module runs an ordinary YOLO11n
``.onnx`` export through :mod:`onnxruntime` (already a project dependency) to
produce equivalent detections from an arbitrary RGB frame.

The output contract mirrors what the emulated camera needs: a list of
:class:`Detected` tuples carrying the box in **normalized ``[0, 1]`` xyxy**
coordinates, a confidence score, and the COCO-80 class **name**.  The name (not
the raw class index) is emitted deliberately — see the label-alignment note on
:meth:`OnnxYoloDetector.detect` — so the emulated IMX500 can re-index it against
whatever label ordering the network intrinsics carry.

Only ``numpy`` / ``cv2`` / ``onnxruntime`` are imported here; nothing Pi-only, so
this runs anywhere.
"""

from typing import List, NamedTuple, Protocol, Tuple

import cv2
import numpy as np

# The 80 COCO class names in the canonical order Ultralytics YOLO models emit
# them (output channel index == this list's index).  Used only to turn a class
# index into a name; the emulated IMX500 re-maps the name onto the intrinsics'
# own label ordering.
COCO_CLASSES: Tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


class Detected(NamedTuple):
    """One object detection from the off-Pi detector.

    Attributes:
        box: Bounding box as ``(x0, y0, x1, y1)`` normalized to ``[0, 1]``
            fractions of the frame width/height.
        score: Confidence score in ``[0, 1]``.
        label: The COCO-80 class name (e.g. ``"bird"``).
    """

    box: Tuple[float, float, float, float]
    score: float
    label: str


class Detector(Protocol):
    """A frame-in, detections-out object detector (the emulated camera's stage 1)."""

    def detect(self, frame: np.ndarray) -> List[Detected]:
        """Return the detected objects for one RGB frame."""


class LetterboxTransform(NamedTuple):
    """The scale + padding applied by :func:`_letterbox`, and its inverse.

    Bundling the mapping parameters into one value object keeps the detection
    decode from carrying a wide argument/local list.

    Attributes:
        ratio: The single scale applied to the frame.
        pad_x: Left padding added to centre the frame in the square canvas.
        pad_y: Top padding added to centre the frame.
        width: Original frame width (for normalization).
        height: Original frame height (for normalization).
    """

    ratio: float
    pad_x: int
    pad_y: int
    width: int
    height: int

    def to_normalized(self, box: np.ndarray) -> Tuple[float, float, float, float]:
        """Map a letterbox-space ``(cx, cy, w, h)`` box to normalized xyxy.

        Undoes the padding + scale, then divides by the original frame size,
        clamping to ``[0, 1]``.
        """
        cx, cy, box_w, box_h = box
        x0 = (cx - box_w / 2 - self.pad_x) / self.ratio
        y0 = (cy - box_h / 2 - self.pad_y) / self.ratio
        x1 = (cx + box_w / 2 - self.pad_x) / self.ratio
        y1 = (cy + box_h / 2 - self.pad_y) / self.ratio
        return (
            float(np.clip(x0 / self.width, 0.0, 1.0)),
            float(np.clip(y0 / self.height, 0.0, 1.0)),
            float(np.clip(x1 / self.width, 0.0, 1.0)),
            float(np.clip(y1 / self.height, 0.0, 1.0)),
        )


def _letterbox(frame: np.ndarray, size: int) -> Tuple[np.ndarray, LetterboxTransform]:
    """Resize an RGB frame to a square ``size`` while preserving aspect ratio.

    The frame is scaled by a single ratio (so it is not distorted) and padded to
    ``size x size`` with grey borders, matching YOLO's expected preprocessing.

    Args:
        frame: The source ``(H, W, 3)`` RGB frame.
        size: The square side length of the network input.

    Returns:
        The letterboxed canvas and the :class:`LetterboxTransform` needed to map
        detections back to the original frame.
    """
    height, width = frame.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, LetterboxTransform(ratio, pad_x, pad_y, width, height)


class OnnxYoloDetector:
    """YOLO11n ONNX object detector run via :mod:`onnxruntime`.

    Loads a standard Ultralytics YOLO11n ``.onnx`` export and decodes its
    ``(1, 84, 8400)`` output (4 box coords + 80 class scores per anchor) into
    :class:`Detected` tuples.
    """

    def __init__(
        self,
        model_path: str,
        *,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> None:
        """Create the ONNX session and read the network input size.

        Args:
            model_path: Path to the YOLO11n ``.onnx`` file.
            conf_threshold: Minimum class confidence to keep a detection.
            iou_threshold: IoU threshold for non-maximum suppression.
        """
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        shape = self._session.get_inputs()[0].shape
        # Input shape is (1, 3, H, W); fall back to 640 when a dim is symbolic.
        self._size = int(shape[2]) if isinstance(shape[2], int) else 640
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> List[Detected]:
        """Detect objects in an RGB frame.

        Preprocesses with a letterbox resize, runs the network, decodes the
        raw output, applies non-maximum suppression, and maps the surviving
        boxes back onto the original frame as normalized ``xyxy`` coordinates.

        Note (label alignment): the returned ``label`` is the COCO class *name*,
        not the raw class index.  The IMX500 ``.rpk`` and this generic export do
        not necessarily share a label ordering, so callers must re-index by name
        against their own label list — emitting the raw index would mis-file a
        detected bird under the wrong class and it would never gate as ``bird``.

        Args:
            frame: The source ``(H, W, 3)`` uint8 RGB frame.

        Returns:
            The detected objects, highest-scoring first.
        """
        canvas, transform = _letterbox(frame, self._size)
        blob = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[
            np.newaxis, ...
        ]
        outputs = self._session.run(None, {self._input_name: blob})[0]

        boxes, scores, class_ids = self._decode(outputs)
        results: List[Detected] = []
        for index in self._nms(boxes, scores):
            class_id = int(class_ids[index])
            if class_id >= len(COCO_CLASSES):
                continue
            norm = transform.to_normalized(boxes[index])
            results.append(Detected(norm, float(scores[index]), COCO_CLASSES[class_id]))
        return results

    def _decode(self, outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode the raw ``(1, 84, 8400)`` output into boxes/scores/class ids.

        Args:
            outputs: The network's first output tensor.

        Returns:
            A ``(boxes, scores, class_ids)`` triple where ``boxes`` are
            ``(cx, cy, w, h)`` in letterbox-input pixels, filtered to the
            configured confidence threshold.
        """
        predictions = np.squeeze(outputs, axis=0).transpose()  # (8400, 84)
        class_scores = predictions[:, 4:]
        confidences = class_scores.max(axis=1)
        mask = confidences >= self._conf_threshold
        return (
            predictions[mask, :4],
            confidences[mask],
            class_scores[mask].argmax(axis=1),
        )

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """Run non-maximum suppression, returning the indices to keep.

        Args:
            boxes: ``(N, 4)`` boxes as ``(cx, cy, w, h)`` in input pixels.
            scores: ``(N,)`` confidence scores.

        Returns:
            The kept box indices (may be empty).
        """
        if len(boxes) == 0:
            return []
        # cv2.dnn.NMSBoxes expects xywh with the top-left corner, not the centre.
        xywh = [
            [float(cx - box_w / 2), float(cy - box_h / 2), float(box_w), float(box_h)]
            for cx, cy, box_w, box_h in boxes
        ]
        indices = cv2.dnn.NMSBoxes(
            xywh, scores.tolist(), self._conf_threshold, self._iou_threshold
        )
        if len(indices) == 0:
            return []
        return [int(i) for i in np.array(indices).reshape(-1)]
