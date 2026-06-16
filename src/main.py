"""Main entry point for bird detection and classification application."""

import argparse
import logging
import sys
import threading
import time

import libcamera
from picamera2 import Picamera2  # type: ignore
from picamera2.devices import IMX500  # type: ignore
from picamera2.devices.imx500 import NetworkIntrinsics  # type: ignore

from object_detection import (
    parse_detections,
    get_labels,
)
from classification_pipeline import (
    process_detections,
    setup_classifier,
    ClassificationManager,
    update_detection_classifications_cache,
)
from tracking import StableDetectionTracker
import classification_pipeline

from db.database import make_engine, init_db, make_session_factory
from db.writer import DetectionWriter


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


def wait_for_camera(model_path: str, retry_interval: float = 30.0) -> IMX500:
    """Initialize the IMX500 device, retrying until the camera becomes available.

    The IMX500 constructor raises ``RuntimeError`` when the camera dev-node is
    missing (e.g. the camera is unplugged, mis-seated, or the container lacks
    device access). Rather than letting the process crash and spam full
    tracebacks under the container's restart policy, log a concise warning and
    retry. The detector stays alive and recovers automatically when the camera
    reappears, while the independent API service keeps serving stored images.

    Args:
        model_path: Path to the IMX500 detection network (``.rpk``) firmware.
        retry_interval: Seconds to wait between initialization attempts.

    Returns:
        An initialized :class:`IMX500` instance once the camera is available.
    """
    logger = logging.getLogger("tracking")
    while True:
        try:
            return IMX500(model_path)
        except RuntimeError as exc:
            logger.warning(
                "Camera not available (%s). Retrying in %.0fs...",
                exc,
                retry_interval,
            )
            time.sleep(retry_interval)


def main():
    """Main application entry point."""
    args = get_args()


    tracking_logger = logging.getLogger("tracking")
    tracking_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    tracking_logger.addHandler(handler)

    # Create the database schema up front (the detector owns all DB writes).
    # Doing this before camera init guarantees the SQLite file and tables exist
    # even when the camera is unavailable, so the read-only API can always serve
    # the site (an empty gallery rather than a 500). The engine is reused below
    # for the DetectionWriter's session factory.
    engine = make_engine()
    init_db(engine)

    # Initialize IMX500 device (must be called before instantiation of Picamera2).
    # Retry gracefully if the camera dev-node is missing so the detector does not
    # crash-loop; the API service serves stored images independently.
    imx500 = wait_for_camera(args.model)
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

    # Set up the SQLite persistence layer.  The detector owns all DB writes;
    # the API mounts the same database read-only.  The schema was already
    # created up front (before camera init) using the same engine, so the API
    # can serve the site even when the camera never comes up.
    detection_writer = DetectionWriter(make_session_factory(engine))

    # Initialize camera and classification manager
    results_lock = threading.Lock()
    manager = ClassificationManager(
        classifier,
        use_multithreading=args.multithread,
        use_stable_track_gating=use_stable_tracks,
        tracker=tracker,
        detection_writer=detection_writer,
    )
    manager.set_results_lock(results_lock)

    picam2 = Picamera2(imx500.camera_num)
    SENSOR_W, SENSOR_H = 4056, 3040
    CROP_W, CROP_H = 900, 900

    ANCHOR_X_FRAC = 4 / 13
    ANCHOR_Y_FRAC = 5 / 10

    crop_x = int(SENSOR_W * ANCHOR_X_FRAC)
    crop_y = int(SENSOR_H * ANCHOR_Y_FRAC)

    # Clamp to sensor bounds
    crop_x = max(0, min(crop_x, SENSOR_W - CROP_W))
    crop_y = max(0, min(crop_y, SENSOR_H - CROP_H))

    config = picam2.create_preview_configuration(
        main={
            "size": (640, 640),
            "format": "RGB888"
        },
        controls={
            "FrameRate": intrinsics.inference_rate,
            "ScalerCrop": (crop_x, crop_y, CROP_W, CROP_H),
        },
        buffer_count=12,
        transform=libcamera.Transform(vflip=True, hflip=True),
    )
    print(f"ScalerCrop = ({crop_x}, {crop_y}, {CROP_W}, {CROP_H})")

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
        # Restrict tracking to detections classified as 'bird' by the detection/segmentation model.
        if use_stable_tracks and tracker is not None:
            tracker.update_frame(
                last_results or [],
                # keep_detection=lambda d: labels[int(d.category)].lower() == "bird",  # type: ignore[attr-defined]
            )

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
            update_detection_classifications_cache(last_results, classification_pipeline.classification_results)
    except KeyboardInterrupt:
        manager.stop()
        detection_writer.stop()


if __name__ == "__main__":
    main()