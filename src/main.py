
import argparse
import sys
from functools import lru_cache
from datetime import datetime
import os
import threading
from queue import Queue

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,)
from picamera2.previews import DrmPreview

from classification import Classifier, ONNXClassifier, build_preprocessing

last_detections = []
classification_queue = Queue()
classification_results = {}
results_lock = threading.Lock()


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


def run_bird_classification(image):
    # Placeholder for bird classification logic
    # This function should return the classification result and confidence score
    return classifier.classify(image)


def classification_worker():
    """Worker thread that processes images from the classification queue."""
    while True:
        try:
            item = classification_queue.get(timeout=1)
            if item is None:  # Sentinel value to stop the thread
                break
            
            image, detection_id, detection, labels, classifier_class = item
            
            x, y, w, h = detection.box
            roi = image[y:y+h, x:x+w]

            species, confidence = run_bird_classification(roi)

            image_with_boxes = draw_boxes(image.copy(), detection, labels, species, confidence)

            if classifier_class.lower() == "bird" and species and confidence > 0.4:
                time = datetime.now()
                os.makedirs(f"/home/stefan/Pictures/bird_detections/{species}/", exist_ok=True)

                output_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"/home/stefan/Pictures/bird_detections/{species}/{time}.png", output_image)

            
            with results_lock:
                classification_results[detection_id] = (species, confidence)
            
            classification_queue.task_done()
        except:
            pass




def draw_boxes(image_array, detection, labels, species=None, confidence=None):
    """Draw detection boxes on image array (not on MappedArray)."""
    x, y, w, h = detection.box
    # Create a copy of the array to draw the background with opacity
    overlay = image_array.copy()

    label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

    if confidence is not None:
        label += f" - {confidence:.2f}"
    if species:
        label += f" - {species}"

    # Calculate text size and position
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_x = x + 5
    text_y = y + 15
    # Draw the background rectangle on the overlay
    cv2.rectangle(overlay,
                    (text_x, text_y - text_height),
                    (text_x + text_width, text_y + baseline),
                    (255, 255, 255),  # Background color (white)
                    cv2.FILLED)

    alpha = 0.30
    cv2.addWeighted(overlay, alpha, image_array, 1 - alpha, 0, image_array)

    # Draw detection box
    cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)
    
    # Draw text on top of the background
    cv2.putText(image_array, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return image_array



def process_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    
    detections = last_results
    if detections is None:
        return
    
    labels = get_labels()
    
    with MappedArray(request, stream) as m:
        for detection_id, detection in enumerate(detections):
            # Extract the region of interest as a numpy array
            full_img = m.array.copy()
            classifier_class = labels[int(detection.category)]
            if classifier_class.lower() == "bird":
                # Add image data to queue for classification on another thread
                # Only pass what's needed: image, detection_id, detection object, labels, class
                classification_queue.put((full_img, detection_id, detection, labels, classifier_class))

    



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--bbox-order", choices=["yx", "xy"], default="yx",
                        help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")

    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str,
                        help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
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
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
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

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    # picam2.start_preview(DrmPreview())

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    # Start classification worker thread
    worker_thread = threading.Thread(target=classification_worker, daemon=True)
    worker_thread.start()

    last_results = None
    picam2.pre_callback = process_detections
    try:
        while True:
            last_results = parse_detections(picam2.capture_metadata())
    except KeyboardInterrupt:
        # Stop the worker thread gracefully
        classification_queue.put(None)