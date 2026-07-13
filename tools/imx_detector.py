#!/usr/bin/env python3
"""Decoder for the IMX-format YOLO11n ONNX built by ``tools/build_imx_emulator.py``.

Ultralytics' IMX export form (:class:`FXModel`) emits the same layout the real
``yolo11n_pp.rpk`` produces — a ``(1, N, 4)`` boxes tensor (``cx, cy, w, h`` in
network-input pixels) and a ``(1, N, 80)`` sigmoid class-scores tensor — rather
than the raw ``(1, 84, 8400)`` of a generic export. :class:`ImxOnnxDetector`
decodes that into the same :class:`dev.emulation.yolo.Detected` tuples the rest of
the diagnostics use, so the float / int8 IMX models drop straight into the probe.

Preprocessing (letterbox → ``/255`` → RGB → NCHW) and the box→frame mapping reuse
the off-Pi detector's helpers, so only the network weights differ across models.

Runs on the project ``.venv`` (onnxruntime + cv2 only).
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Block disable so black re-wrapping can't drift the suppression off the anchor line.
# pylint: disable=wrong-import-position
from dev.emulation.yolo import COCO_CLASSES, Detected, _letterbox

# pylint: enable=wrong-import-position


class ImxOnnxDetector:
    """Runs an IMX-format (boxes + scores) YOLO11n ONNX and decodes detections.

    Mirrors :class:`dev.emulation.yolo.OnnxYoloDetector`'s interface (``detect`` →
    ``list[Detected]`` with normalized ``xyxy`` boxes and COCO class names) so it is
    interchangeable in the diagnostics, but decodes the two-tensor IMX output
    instead of the raw ``(1, 84, 8400)`` tensor.
    """

    def __init__(
        self,
        model_path: str,
        *,
        conf_threshold: float = 0.01,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        """Create the ONNX session.

        Args:
            model_path: Path to the IMX-format ``.onnx`` (from build_imx_emulator).
            conf_threshold: Minimum class confidence to keep a detection.
            iou_threshold: IoU threshold for non-maximum suppression.
            imgsz: Network input side length (must match how the model was exported).
        """
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._imgsz = imgsz
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> List[Detected]:
        """Detect objects in an RGB frame.

        Args:
            frame: The source ``(H, W, 3)`` uint8 RGB frame.

        Returns:
            The detected objects (normalized ``xyxy`` box, score, COCO name),
            highest-scoring first is not guaranteed (NMS order).
        """
        canvas, transform = _letterbox(frame, self._imgsz)
        blob = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[
            np.newaxis, ...
        ]
        outputs = self._session.run(None, {self._input_name: blob})
        boxes, scores = self._find_boxes_scores(outputs)
        if boxes is None or scores is None:
            return []

        conf = scores.max(axis=1)
        class_ids = scores.argmax(axis=1)
        mask = conf >= self._conf_threshold
        boxes, conf, class_ids = boxes[mask], conf[mask], class_ids[mask]

        results: List[Detected] = []
        for index in self._nms(boxes, conf):
            class_id = int(class_ids[index])
            if class_id >= len(COCO_CLASSES):
                continue
            norm = transform.to_normalized(boxes[index])
            results.append(Detected(norm, float(conf[index]), COCO_CLASSES[class_id]))
        return results

    @staticmethod
    def _find_boxes_scores(
        outputs: List[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Pick the ``(N, 4)`` boxes and ``(N, 80)`` scores tensors by shape.

        The export emits several tensors; select them by their trailing dimension
        so the decode is robust to output ordering.

        Args:
            outputs: The raw ONNX session outputs.

        Returns:
            ``(boxes, scores)`` as ``(N, 4)`` / ``(N, 80)`` arrays, or ``None`` when
            a matching tensor is absent.
        """
        boxes: Optional[np.ndarray] = None
        scores: Optional[np.ndarray] = None
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes = arr[0]
            elif arr.ndim == 3 and arr.shape[-1] == 80:
                scores = arr[0]
        return boxes, scores

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """Run non-maximum suppression, returning the kept indices.

        Args:
            boxes: ``(N, 4)`` boxes as ``(cx, cy, w, h)`` in input pixels.
            scores: ``(N,)`` confidence scores.

        Returns:
            The kept box indices (may be empty).
        """
        if len(boxes) == 0:
            return []
        xywh = [
            [float(cx - w / 2), float(cy - h / 2), float(w), float(h)]
            for cx, cy, w, h in boxes
        ]
        indices = cv2.dnn.NMSBoxes(
            xywh, scores.tolist(), self._conf_threshold, self._iou_threshold
        )
        if len(indices) == 0:
            return []
        return [int(i) for i in np.array(indices).reshape(-1)]
