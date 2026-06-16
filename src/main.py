"""Main entry point for bird detection and classification application."""

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
from config import config as app_config

from db.database import make_engine, init_db, make_session_factory
from db.writer import DetectionWriter


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
    tracking_logger = logging.getLogger("tracking")
    tracking_logger.setLevel(logging.DEBUG if app_config.debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    tracking_logger.addHandler(handler)

    # Create the database schema up front (the detector owns all DB writes).
    engine = make_engine()
    init_db(engine)

    # Initialize IMX500 device (must be called before instantiation of Picamera2).
    imx500 = wait_for_camera(app_config.model)
    intrinsics = imx500.network_intrinsics

    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    # Override intrinsics from config
    for key, value in vars(app_config).items():
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

    if app_config.print_intrinsics:
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
    # Prefer config.fps if supplied; fall back to intrinsics inference rate.
    fps = app_config.fps or int(getattr(intrinsics, "inference_rate", 0) or 0) or 1
    if (
        app_config.object_duration_threshold
        and app_config.object_duration_threshold > 0
    ):
        min_stable_frames = max(
            1, int((app_config.object_duration_threshold * fps) + 0.9999)
        )
        use_stable_tracks = True

        from track_logging import TrackingLogger

        logger = TrackingLogger()

        tracker = StableDetectionTracker(
            iou_threshold=0.6,
            min_stable_frames=min_stable_frames,
            # Reasonable default; can be promoted to a config field later.
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
        use_multithreading=app_config.multithread,
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
        main={"size": (640, 640), "format": "RGB888"},
        controls={
            "FrameRate": intrinsics.inference_rate,
            "ScalerCrop": (crop_x, crop_y, CROP_W, CROP_H),
        },
        buffer_count=12,
        transform=libcamera.Transform(vflip=True, hflip=True),
    )
    print(f"ScalerCrop = ({crop_x}, {crop_y}, {CROP_W}, {CROP_H})")

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=app_config.preview)

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
                app_config.threshold,
                picam2,
            )
            # Update the detection classifications cache for temporal filtering
            update_detection_classifications_cache(
                last_results, classification_pipeline.classification_results
            )
    except KeyboardInterrupt:
        manager.stop()
        detection_writer.stop()


if __name__ == "__main__":
    main()
