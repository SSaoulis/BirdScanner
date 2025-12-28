"""Main entry point for bird detection and classification application."""

import argparse
import logging
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
    StableDetectionTracker,
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
    parser.add_argument(
        "--object-duration-threshold",
        dest="object_duration_threshold",
        type=float,
        default=0.2,
        help=(
            "Total time (seconds) a track must be stable (IoU>0.6 across frames) before running bird "
            "classification. Set to 0 to revert to legacy per-frame logic."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for track lifecycle events",
    )
    # add flag to turn on preview
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Enable camera preview window",
    )
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = get_args()


    tracking_logger = logging.getLogger("tracking")
    tracking_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    tracking_logger.addHandler(handler)

    
    

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

    # Configure new stable-track gating.
    # - If duration == 0 => revert to legacy behaviour.
    # - Otherwise compute min_stable_frames from duration * fps (ceil, min 1).
    # Prefer args.fps if supplied; fall back to intrinsics inference rate.
    fps = args.fps or int(getattr(intrinsics, "inference_rate", 0) or 0) or 1
    if args.object_duration_threshold and args.object_duration_threshold > 0:
        min_stable_frames = max(1, int((args.object_duration_threshold * fps) + 0.9999))
        use_stable_tracks = True

        from track_logging import TrackingLogger
        logger = TrackingLogger()

        tracker = StableDetectionTracker(
            iou_threshold=0.6,
            min_stable_frames=min_stable_frames,
            # Reasonable default; can be promoted to a CLI arg later.
            max_missing_frames=max(1, int(1.0 * fps)),
            on_track_became_stable=logger.log_stable_track,
            on_track_deleted=logger.log_deleted_track,
        )
    else:
        use_stable_tracks = False
        tracker = None

    # Initialize camera and classification manager
    results_lock = threading.Lock()
    manager = ClassificationManager(
        classifier,
        use_multithreading=args.multithread,
        use_stable_track_gating=use_stable_tracks,
        tracker=tracker,
    )
    manager.set_results_lock(results_lock)

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12,
    )
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=args.preview)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    # Get labels for display
    labels = get_labels(intrinsics)

    # Set up detection processing callback
    last_results = None

    def detection_callback(request):
        """Callback for processing detections on each frame."""
        nonlocal last_results

        # Update tracker state once per frame (before enqueuing items).
        if use_stable_tracks and tracker is not None:
            tracker.update_frame(last_results or [])

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