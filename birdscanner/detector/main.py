"""Main entry point for bird detection and classification application.

``main`` orchestrates a sequence of focused builders — logging, database,
network intrinsics, stable-track gating, the classification manager, the camera,
and the control server — then hands off to the capture loop. Each builder owns
one slice of startup so the entry point stays a short, readable script.
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
    PipelineContext,
    update_detection_classifications_cache,
)
from birdscanner.ml.tracking import (
    StableDetectionTracker,
    StableTrack,
    stable_detection_tracker,
)
from birdscanner.ml.best_frame import BestFrameSelector
from birdscanner.ml import classification_pipeline
from birdscanner.detector.camera_server import camera_server_port, start_camera_server
from birdscanner.detector.video_recorder import VideoRecorder
from birdscanner.detector.config import config as app_config
from birdscanner.detector.crop import (
    SENSOR_W,
    SENSOR_H,
    SensorDimensions,
    crop_config_path,
    default_crop_region,
    load_crop_region,
    main_stream_size_for_crop,
)
from birdscanner.detector.crop_controller import CropController, CropControllerConfig
from birdscanner.detector.paths import (
    class_to_idx_path,
    classifier_model_path,
    coco_labels_path,
)
from birdscanner.db.database import make_engine, init_db, make_session_factory
from birdscanner.db.writer import DetectionWriter
from birdscanner.db.deleter import delete_detection

logger = logging.getLogger("tracking")


@dataclass
class _Gating:
    """Result of building the stable-track gating machinery.

    Attributes:
        use_stable_tracks: Whether stable-track gating is active.
        tracker: The stability tracker; the module-global default in legacy
            per-frame mode (where it is unused).
        best_frame_selector: Per-track best-frame store, or ``None`` when gating
            is off.
        video_recorder: The clip recorder, or ``None`` when gating is off or
            video is disabled.
    """

    use_stable_tracks: bool
    tracker: StableDetectionTracker
    best_frame_selector: Optional[BestFrameSelector]
    video_recorder: Optional[VideoRecorder]


@dataclass
class _Camera:
    """The started camera and the objects needed to drive/reconfigure it.

    Attributes:
        picam2: The started ``Picamera2`` instance.
        imx500: The initialised IMX500 device.
        intrinsics: The (overridden) network intrinsics.
        crop_controller: Owns live crop changes and the shared camera lock.
    """

    picam2: Any
    imx500: Any
    intrinsics: Any
    crop_controller: CropController


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


def _configure_logging() -> None:
    """Configure the ``tracking`` logger to stdout at the configured level."""
    logger.setLevel(logging.DEBUG if app_config.debug else logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


def _prepare_intrinsics(imx500: IMX500) -> Any:
    """Return the network intrinsics with the config overrides applied.

    Validates that the network is an object-detection task, pushes each
    non-``None`` override from ``config.intrinsics`` onto the intrinsics object,
    loads default labels when none were supplied, and (when
    ``config.print_intrinsics`` is set) prints them and exits.

    Args:
        imx500: The initialised IMX500 device.

    Returns:
        The prepared network intrinsics object.
    """
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        sys.exit()

    for key, value in vars(app_config.intrinsics).items():
        if key == "labels" and value is not None:
            with open(value, "r", encoding="utf-8") as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    if intrinsics.labels is None:
        with open(coco_labels_path(), "r", encoding="utf-8") as f:
            intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()

    if app_config.print_intrinsics:
        print(intrinsics)
        sys.exit()

    return intrinsics


def _min_stable_frames(fps: int) -> int:
    """Frames a track must be stable before classification, from duration × fps."""
    return max(1, int((app_config.object_duration_threshold * fps) + 0.9999))


def _build_gating(intrinsics: Any) -> _Gating:
    """Build the stable-track gating machinery from the config and intrinsics.

    When ``object_duration_threshold <= 0`` gating is disabled (legacy per-frame
    mode) and every returned component is ``None``. Otherwise this constructs the
    tracker (wired to log lifecycle events and free per-track best frames), the
    best-frame selector, and — when video is enabled — the clip recorder.

    Args:
        intrinsics: The prepared network intrinsics (for the inference rate).

    Returns:
        The populated :class:`_Gating` bundle.
    """
    if not app_config.object_duration_threshold > 0:
        return _Gating(False, stable_detection_tracker, None, None)

    from birdscanner.detector.track_logging import TrackingLogger

    fps = (
        app_config.intrinsics.fps
        or int(getattr(intrinsics, "inference_rate", 0) or 0)
        or 1
    )
    track_logger = TrackingLogger()
    selector = BestFrameSelector()

    def on_track_deleted(track: StableTrack) -> None:
        """Log the deletion and release the track's retained best frame."""
        track_logger.log_deleted_track(track)
        selector.discard(track.track_id)

    tracker = StableDetectionTracker(
        iou_threshold=0.6,
        min_stable_frames=_min_stable_frames(fps),
        max_missing_frames=max(1, int(1.0 * fps)),
        on_track_became_stable=track_logger.log_stable_track,
        on_track_deleted=on_track_deleted,
    )

    recorder = (
        VideoRecorder(
            fps=fps,
            pre_roll_seconds=app_config.video.pre_roll_seconds,
            post_roll_seconds=app_config.video.post_roll_seconds,
        )
        if app_config.video.save
        else None
    )
    return _Gating(True, tracker, selector, recorder)


def _build_manager(
    classifier: Any, gating: _Gating, detection_writer: DetectionWriter
) -> ClassificationManager:
    """Assemble the :class:`ClassificationManager` and its pipeline context.

    Args:
        classifier: The species classifier.
        gating: The stable-track gating bundle.
        detection_writer: The background DB writer.

    Returns:
        A ready manager with its results lock installed.
    """
    recorder = gating.video_recorder
    context = PipelineContext(
        classifier=classifier,
        tracker=gating.tracker,
        detection_writer=detection_writer,
        best_frame_selector=gating.best_frame_selector,
        record_fn=recorder.trigger if recorder is not None else None,
        video_frame_fn=recorder.add_frame if recorder is not None else None,
    )
    manager = ClassificationManager(
        context,
        use_multithreading=app_config.multithread,
        use_stable_track_gating=gating.use_stable_tracks,
    )
    manager.set_results_lock(threading.Lock())
    return manager


def _build_camera(imx500: IMX500, intrinsics: Any) -> _Camera:
    """Start the camera at the persisted crop and wrap it in a :class:`_Camera`.

    Loads the saved detection crop (falling back to the feeder default), sizes
    the ``main`` stream to its aspect ratio, starts the camera, and builds the
    :class:`CropController` that owns subsequent live crop changes.

    Args:
        imx500: The initialised IMX500 device.
        intrinsics: The prepared network intrinsics.

    Returns:
        The started camera bundle.
    """
    picam2 = Picamera2(imx500.camera_num)
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

    crop_controller = CropController(
        picam2,
        CropControllerConfig(
            region=crop_region,
            main_size=initial_main_size,
            config_factory=build_camera_config,
            config_path=crop_config_path(),
            sensor=SensorDimensions(SENSOR_W, SENSOR_H),
        ),
    )
    return _Camera(picam2, imx500, intrinsics, crop_controller)


def _start_control_server(camera: _Camera, engine: Any) -> Optional[Any]:
    """Start the auxiliary control server (snapshots + crop + detection deletes).

    The read-only API proxies here to surface a live test image, the crop editor,
    and detection deletes (the detector owns the camera + read-write data volume
    exclusively). If the port is already in use, log a warning and continue
    without it rather than killing the detection pipeline.

    Args:
        camera: The started camera bundle.
        engine: The database engine (for the delete handler).

    Returns:
        The running server, or ``None`` when the port could not be bound.
    """
    image_root = Path(classification_pipeline.IMAGE_DIR)
    delete_session_factory = make_session_factory(engine)

    def handle_delete(detection_id: int) -> bool:
        """Delete a detection by id; returns True if a record existed."""
        return delete_detection(delete_session_factory, image_root, detection_id)

    snapshot_port = camera_server_port()
    try:
        return start_camera_server(
            camera.picam2,
            snapshot_port,
            crop_controller=camera.crop_controller,
            delete_detection=handle_delete,
        )
    except OSError as exc:
        logger.warning(
            "Camera snapshot server could not bind port %d (%s); continuing "
            "without it. Set CAMERA_SERVER_PORT to a free port to enable the "
            "Camera tab's Test Camera button.",
            snapshot_port,
            exc,
        )
        return None


def _run_capture_loop(
    camera: _Camera, manager: ClassificationManager, gating: _Gating
) -> None:
    """Run the frame-capture loop until interrupted.

    Installs the per-frame detection callback (which updates the tracker and
    queues bird detections) and then repeatedly captures metadata, parses
    detections under the crop controller's lock, and refreshes the legacy
    temporal-filter cache.

    Args:
        camera: The started camera bundle.
        manager: The classification manager the callback feeds.
        gating: The stable-track gating bundle.
    """
    labels = get_labels(camera.intrinsics)
    state: dict = {"last_results": None}

    def detection_callback(request):
        """Update tracker state and process detections for one frame."""
        if gating.use_stable_tracks:
            gating.tracker.update_frame(state["last_results"] or [])
        process_detections(request, "main", state["last_results"], manager, labels)

    camera.picam2.pre_callback = detection_callback

    while True:
        # Hold the crop controller's lock across capture + parse so a UI-triggered
        # crop reconfigure (which may stop/start the camera) can never interleave
        # with an in-flight capture.
        with camera.crop_controller.camera_lock:
            metadata = camera.picam2.capture_metadata()
            state["last_results"] = parse_detections(
                metadata,
                camera.imx500,
                camera.intrinsics,
                app_config.threshold,
                camera.picam2,
            )
        update_detection_classifications_cache(
            state["last_results"], classification_pipeline.classification_results
        )


def _shutdown(
    control_server: Optional[Any],
    manager: ClassificationManager,
    detection_writer: DetectionWriter,
) -> None:
    """Stop the control server, classification worker, and DB writer."""
    if control_server is not None:
        control_server.shutdown()
    manager.stop()
    detection_writer.stop()


def main() -> None:
    """Bird detection and classification application entry point."""
    _configure_logging()

    # Create the database schema up front (the detector owns all DB writes).
    engine = make_engine()
    init_db(engine)

    # IMX500 must be initialised before Picamera2.
    imx500 = wait_for_camera(app_config.model)
    intrinsics = _prepare_intrinsics(imx500)

    classifier = setup_classifier(
        str(classifier_model_path()), str(class_to_idx_path())
    )
    gating = _build_gating(intrinsics)

    # The detector owns all DB writes; the API mounts the same database read-only.
    detection_writer = DetectionWriter(make_session_factory(engine))
    manager = _build_manager(classifier, gating, detection_writer)

    camera = _build_camera(imx500, intrinsics)
    control_server = _start_control_server(camera, engine)

    try:
        _run_capture_loop(camera, manager, gating)
    except KeyboardInterrupt:
        _shutdown(control_server, manager, detection_writer)


if __name__ == "__main__":
    main()
