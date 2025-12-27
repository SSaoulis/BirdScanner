import argparse
import sys
from functools import lru_cache
from datetime import datetime
import os
import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2 # type: ignore
from picamera2.devices import IMX500 # type: ignore
from picamera2.devices.imx500 import (NetworkIntrinsics,) # type: ignore
from classification import Classifier, ONNXClassifier, build_preprocessing

last_detections = []
classification_results = {}
last_bird_classification = None  # (box, species, confidence) for temporal filtering
OUTPUT_DIR = "/home/stefan/Pictures/bird_detections"

MODEL_PATH = "local/convnext_v2_tiny.onnx"
CLASS_TO_IDX_PATH = "assets/convnext_v2_tiny.onnx_class_to_idx.json"

onnx_model = ONNXClassifier(str(MODEL_PATH))
preprocessing = build_preprocessing({
    "size": (384, 384),
    "rgb_values": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "center_crop": 1.0,
    "simple_crop": False,
})
classifier = Classifier(onnx_model, CLASS_TO_IDX_PATH, preprocessing=preprocessing)


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes.
    Boxes are in format (x, y, w, h).
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
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
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
    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels
    labels = [label for label in labels if label and label != "-"]
    return labels


def preprocess_roi(image, box):
    """
    Preprocess the region of interest by:
    1. Padding it to a square
    2. Expanding by 10% through the center
    3. Clamping to image boundaries while maintaining square shape
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


def run_bird_classification(image):
    return classifier.classify(image)


def draw_boxes(image_array, coords, detection, labels, species=None, confidence=None):
    """Draw detection boxes on image array (not on MappedArray)."""
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


def process_single_detection(item, *, results_lock):
    """Process one detection item (sync or async depending on manager)."""
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
        species, confidence = run_bird_classification(roi)
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


class ClassificationManager:
    def __init__(self, *, use_multithreading: bool = False, queue_maxsize: int = 0):
        self.use_multithreading = use_multithreading
        self._results_lock = None
        self._queue = None
        self._thread = None
        self._stop_event = None

        if self.use_multithreading:
            import threading
            from queue import Queue

            self._stop_event = threading.Event()
            self._queue = Queue(maxsize=queue_maxsize)
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    def set_results_lock(self, results_lock):
        self._results_lock = results_lock

    def process(self, item):
        if not self.use_multithreading:
            process_single_detection(item, results_lock=self._results_lock)
            return

        from queue import Full

        try:
            self._queue.put_nowait(item) # type: ignore


        except Full:
            # Drop frame if queue is full.
            return

    def _worker_loop(self):
        while not self._stop_event.is_set():# type: ignore
            item = self._queue.get()# type: ignore
            if item is None:
                self._queue.task_done()# type: ignore
                break
            process_single_detection(item, results_lock=self._results_lock)
            self._queue.task_done()# type: ignore

    def stop(self):
        if not self.use_multithreading:
            return
        self._stop_event.set()# type: ignore
        self._queue.put(None)# type: ignore
        self._thread.join(timeout=5)# type: ignore


def process_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""

    detections = last_results
    if detections is None:
        return

    labels = get_labels()

    with MappedArray(request, stream) as m:
        full_img = m.array.copy()
        for detection_id, detection in enumerate(detections):
            _, coords = preprocess_roi(full_img, detection.box)
            image_with_boxes = draw_boxes(full_img.copy(), coords, detection, labels)
            m.array[:] = image_with_boxes

            classifier_class = labels[int(detection.category)]
            if classifier_class.lower() == "bird":
                manager.process((full_img, detection_id, detection, labels, classifier_class))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Path of the model",
        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
    )
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument(
        "--bbox-order",
        choices=["yx", "xy"],
        default="yx",
        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)",
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument(
        "-r",
        "--preserve-aspect-ratio",
        action=argparse.BooleanOptionalAction,
        help="preserve the pixel aspect ratio of the input tensor",
    )
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true", help="Print JSON network_intrinsics then exit")
    parser.add_argument(
        "--multithread",
        action="store_true",
        help="Enable background processing thread for classification",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == "labels" and value is not None:
            with open(value, "r") as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()
    if args.print_intrinsics:
        print(intrinsics)
        exit()

    import threading

    results_lock = threading.Lock()

    manager = ClassificationManager(use_multithreading=args.multithread)
    manager.set_results_lock(results_lock)

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    # picam2.start_preview(DrmPreview())
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = process_detections
    try:
        while True:
            last_results = parse_detections(picam2.capture_metadata())
    except KeyboardInterrupt:
        manager.stop()