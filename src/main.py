"""Main entry point for bird detection and classification application."""

import argparse
import sys
import threading

from picamera2 import Picamera2  # type: ignore
from picamera2.devices import IMX500  # type: ignore
from picamera2.devices.imx500 import NetworkIntrinsics  # type: ignore

from object_detection import (
    parse_detections,
    get_labels,
    process_detections,
    setup_classifier,
    ClassificationManager,
    update_detection_classifications_cache,
)
import object_detection


def get_args():
    """Parse command-line arguments."""
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


def main():
    """Main application entry point."""
    args = get_args()

    # Initialize IMX500 device (must be called before instantiation of Picamera2)
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

    # Load default labels if not provided
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()
    
    if args.print_intrinsics:
        print(intrinsics)
        exit()

    # Initialize classifier
    classifier = setup_classifier(
        "local/convnext_v2_tiny_int8.onnx",
        "assets/convnext_v2_tiny.onnx_class_to_idx.json",
    )

    # Initialize camera and classification manager
    results_lock = threading.Lock()
    manager = ClassificationManager(classifier, use_multithreading=args.multithread)
    manager.set_results_lock(results_lock)

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12,
    )
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    # Get labels for display
    labels = get_labels(intrinsics)

    # Set up detection processing callback
    last_results = None

    def detection_callback(request):
        """Callback for processing detections on each frame."""
        nonlocal last_results
        process_detections(request, "main", last_results, manager, labels)

    picam2.pre_callback = detection_callback

    # Main loop
    try:
        while True:
            last_results = parse_detections(
                picam2.capture_metadata(),
                imx500,
                intrinsics,
                args.threshold,
                picam2,
            )
            # Update the detection classifications cache for temporal filtering
            update_detection_classifications_cache(last_results, object_detection.classification_results)
    except KeyboardInterrupt:
        manager.stop()


if __name__ == "__main__":
    main()