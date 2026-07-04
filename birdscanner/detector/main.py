"""Main entry point for bird detection and classification application."""

import logging
import sys
import threading
import time

import libcamera
from picamera2 import Picamera2  # type: ignore
from picamera2.devices import IMX500  # type: ignore
from picamera2.devices.imx500 import NetworkIntrinsics  # type: ignore

from birdscanner.ml.object_detection import (
    parse_detections,
    get_labels,
)
from birdscanner.ml.classification_pipeline import (
    process_detections,
    setup_classifier,
    ClassificationManager,
    update_detection_classifications_cache,
)
from birdscanner.ml.tracking import StableDetectionTracker, StableTrack
from birdscanner.ml.best_frame import BestFrameSelector
from birdscanner.ml import classification_pipeline
from birdscanner.detector.camera_server import camera_server_port, start_camera_server
from birdscanner.detector.video_recorder import VideoRecorder
from birdscanner.detector.config import config as app_config
from birdscanner.detector.crop import (
    SENSOR_W,
    SENSOR_H,
    crop_config_path,
    default_crop_region,
    load_crop_region,
    main_stream_size_for_crop,
)
from birdscanner.detector.crop_controller import CropController
from birdscanner.detector.paths import (
    class_to_idx_path,
    classifier_model_path,
    coco_labels_path,
)
from birdscanner.db.database import make_engine, init_db, make_session_factory
from birdscanner.db.writer import DetectionWriter
from birdscanner.db.deleter import delete_detection


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
        sys.exit()

    # Override intrinsics from config
    for key, value in vars(app_config).items():
        if key == "labels" and value is not None:
            with open(value, "r", encoding="utf-8") as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Load default labels if not provided
    if intrinsics.labels is None:
        with open(coco_labels_path(), "r", encoding="utf-8") as f:
            intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()

    if app_config.print_intrinsics:
        print(intrinsics)
        sys.exit()

    # Initialize classifier
    classifier = setup_classifier(
        str(classifier_model_path()),
        str(class_to_idx_path()),
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

        from birdscanner.detector.track_logging import TrackingLogger

        logger = TrackingLogger()

        # Retains the best frame per track for the saved still/thumbnail; freed
        # when a track ends (see the on_track_deleted callback below). A
        # non-optional local is captured by the closure so it never dereferences
        # a possibly-None value.
        selector = BestFrameSelector()
        best_frame_selector: BestFrameSelector | None = selector

        def on_track_deleted(track: StableTrack) -> None:
            """Log the deletion and release the track's retained best frame."""
            logger.log_deleted_track(track)
            selector.discard(track.track_id)

        tracker = StableDetectionTracker(
            iou_threshold=0.6,
            min_stable_frames=min_stable_frames,
            # Reasonable default; can be promoted to a config field later.
            max_missing_frames=max(1, int(1.0 * fps)),
            on_track_became_stable=logger.log_stable_track,
            on_track_deleted=on_track_deleted,
        )

        # Records a short mp4 clip around each saved detection from an in-RAM
        # pre-roll buffer (Pi 5 has no hardware encoder, so this stays off-thread
        # and only encodes on a detection).
        video_recorder: VideoRecorder | None = (
            VideoRecorder(
                fps=fps,
                pre_roll_seconds=app_config.video_pre_roll_seconds,
                post_roll_seconds=app_config.video_post_roll_seconds,
            )
            if app_config.save_video
            else None
        )
    else:
        use_stable_tracks = False
        tracker = None
        best_frame_selector = None
        video_recorder = None

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
        best_frame_selector=best_frame_selector,
        record_fn=video_recorder.trigger if video_recorder is not None else None,
        video_frame_fn=(
            video_recorder.add_frame if video_recorder is not None else None
        ),
    )
    manager.set_results_lock(results_lock)

    picam2 = Picamera2(imx500.camera_num)

    # The detection crop region is variable and configured from the UI. Load the
    # last saved region (falling back to the historical 900x900 feeder default)
    # and size the ``main`` stream to its aspect ratio so the region is not
    # stretched into a square frame.
    crop_region = load_crop_region(
        crop_config_path(), default_crop_region(SENSOR_W, SENSOR_H)
    )
    initial_main_size = main_stream_size_for_crop(crop_region.w, crop_region.h)

    def build_camera_config(
        main_size: tuple[int, int], scaler_crop: tuple[int, int, int, int]
    ):
        """Build the preview configuration for a given main size and crop.

        Centralises every picamera2-specific knob so :class:`CropController` can
        rebuild an equivalent configuration when an aspect-ratio change forces a
        ``main`` stream resize.

        Args:
            main_size: The ``(w, h)`` of the ISP ``main`` output stream.
            scaler_crop: The ``(x, y, w, h)`` sensor ScalerCrop region.

        Returns:
            A picamera2 preview configuration object.
        """
        return picam2.create_preview_configuration(
            # picamera2's "888" format names are byte-reversed vs. the numpy
            # array they yield: "BGR888" delivers an [R, G, B]-ordered array. The
            # whole pipeline (ConvNeXt classifier, PIL thumbnails, the cv2
            # RGB2BGR writes) assumes RGB, so we must request BGR888 to actually
            # get RGB. Using "RGB888" here yields BGR and swaps red<->blue
            # everywhere downstream.
            main={"size": main_size, "format": "BGR888"},
            controls={
                "FrameRate": intrinsics.inference_rate,
                "ScalerCrop": scaler_crop,
            },
            # 6 buffers keep ample jitter margin while halving DMA-heap pressure
            # vs the inherited 12. Inside the container the kernel CMA pool is
            # shared with the IMX500 firmware upload and the 2028x1520 raw sensor
            # stream (the dominant consumer, fixed at the sensor's smallest mode),
            # so 12 buffers exhausted CMA and crashed picam2.start() with ENOMEM.
            buffer_count=6,
            transform=libcamera.Transform(vflip=True, hflip=True),
        )

    config = build_camera_config(initial_main_size, crop_region.as_tuple())
    print(f"ScalerCrop = {crop_region.as_tuple()}  main={initial_main_size}")

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=app_config.preview)

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    # The crop controller owns every live change to the detection region and
    # serialises camera access with the main capture loop below via its lock, so
    # a UI-triggered reconfigure never races an in-flight capture.
    crop_controller = CropController(
        picam2=picam2,
        region=crop_region,
        main_size=initial_main_size,
        config_factory=build_camera_config,
        config_path=crop_config_path(),
        sensor_w=SENSOR_W,
        sensor_h=SENSOR_H,
    )

    # Deletion runs synchronously on the control server thread against the same
    # engine the writer uses, removing the DB row and the image + thumbnail
    # files (the detector owns the read-write data volume; the API proxies here).
    from pathlib import Path

    image_root = Path(classification_pipeline.IMAGE_DIR)
    delete_session_factory = make_session_factory(engine)

    def handle_delete(detection_id: int) -> bool:
        """Delete a detection by id; returns True if a record existed."""
        return delete_detection(delete_session_factory, image_root, detection_id)

    # Expose on-demand snapshots + crop control + detection deletion so the
    # read-only API can surface a live test image, the crop editor, and delete
    # detections (the detector owns the camera exclusively; the API proxies to
    # this). The control server is auxiliary: if its port is already in use
    # (another service or a stale instance), log a warning and run without it
    # rather than killing the detection pipeline.
    snapshot_port = camera_server_port()
    try:
        camera_server = start_camera_server(
            picam2,
            snapshot_port,
            crop_controller=crop_controller,
            delete_detection=handle_delete,
        )
    except OSError as exc:
        tracking_logger.warning(
            "Camera snapshot server could not bind port %d (%s); continuing "
            "without it. Set CAMERA_SERVER_PORT to a free port to enable the "
            "Camera tab's Test Camera button.",
            snapshot_port,
            exc,
        )
        camera_server = None

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
            # Hold the crop controller's lock across capture + parse so a
            # UI-triggered crop reconfigure (which may stop/start the camera)
            # can never interleave with an in-flight capture.
            with crop_controller.camera_lock:
                metadata = picam2.capture_metadata()
                last_results = parse_detections(
                    metadata,
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
        if camera_server is not None:
            camera_server.shutdown()
        manager.stop()
        detection_writer.stop()


if __name__ == "__main__":
    main()
