"""Main entry point for bird detection and classification application.

``main`` reads as a short startup script: configure logging, create the DB
schema, bring up the camera (:mod:`birdscanner.detector.hardware.camera`), wire the
classification pipeline (:mod:`birdscanner.detector.pipeline.gating`), start the control
server, then hand off to :func:`_run_capture_loop`.  The per-frame loop is the
only substantial logic that lives here; the setup steps live in their own
cohesive modules.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

from birdscanner.ml.object_detection import (
    InferenceContext,
    filter_included_detections,
    parse_detections,
    get_labels,
)
from birdscanner.ml.classification_pipeline import (
    process_detections,
    setup_classifier,
    ClassificationManager,
)
from birdscanner.ml import classification_pipeline
from birdscanner.detector.hardware.camera import (
    Camera,
    build_camera,
    prepare_intrinsics,
    wait_for_camera,
)
from birdscanner.detector.hardware.camera_server import (
    ControlServerDeps,
    camera_server_port,
    start_camera_server,
)
from birdscanner.detector.config.config import config as app_config
from birdscanner.detector.pipeline.gating import (
    Gating,
    build_gating,
    build_geo_adjuster,
    build_manager,
)
from birdscanner.detector.pipeline.geo_priors import refresh_geo_priors
from birdscanner.detector.config.settings import load_settings, settings_config_path
from birdscanner.detector.config.settings_controller import (
    SettingsController,
    apply_settings_to_config,
    apply_settings_to_context,
)
from birdscanner.detector.pipeline.track_logging import configure_logging
from birdscanner.detector.paths import (
    class_to_idx_path,
    classifier_model_path,
)
from birdscanner.detector.hardware.raw_frame import build_clip_frame_source
from birdscanner.db.database import make_engine, init_db, make_session_factory
from birdscanner.db.writer import DetectionWriter
from birdscanner.db.deleter import delete_detection
from birdscanner.db.corrector import correct_detection_species
from birdscanner.db.custom_species import add_custom_species, list_custom_species

logger = logging.getLogger("tracking")

# Grace period before the requested restart exits the process, so the HTTP
# response to the restart request is flushed first.
_RESTART_DELAY_SECONDS = 0.5


def _schedule_restart() -> None:
    """Exit the process shortly so Docker's restart policy relaunches it.

    The Settings page's "Apply & restart" button hits this to pick up
    restart-only settings.  A hard ``os._exit`` from a timer thread is used
    (rather than ``sys.exit``, which only unwinds the calling thread) so the
    container actually restarts under the ``restart: unless-stopped`` policy.
    Outside Docker the process simply exits and must be started again.
    """
    logger.info("Restart requested; exiting in %.1fs.", _RESTART_DELAY_SECONDS)
    threading.Timer(
        _RESTART_DELAY_SECONDS,
        lambda: os._exit(0),  # pylint: disable=protected-access
    ).start()


def _start_control_server(
    camera: Camera,
    engine: Any,
    settings_controller: SettingsController,
    species_labels: list[str],
) -> Optional[Any]:
    """Start the auxiliary control server (snapshots + crop + deletes + settings).

    The read-only API proxies here to surface a live test image, the crop editor,
    detection deletes, species corrections, and the Settings page (the detector
    owns the camera + read-write data volume exclusively). If the port is already
    in use, log a warning and continue without it rather than killing the
    detection pipeline.

    Args:
        camera: The started camera bundle.
        engine: The database engine (for the delete + correction handlers).
        settings_controller: Applies + persists runtime settings changes.
        species_labels: The classifier vocabulary served at ``GET /labels`` and
            used to validate species corrections.

    Returns:
        The running server, or ``None`` when the port could not be bound.
    """
    image_root = Path(classification_pipeline.IMAGE_DIR)
    control_session_factory = make_session_factory(engine)

    def handle_delete(detection_id: int) -> bool:
        """Delete a detection by id; returns True if a record existed."""
        return delete_detection(control_session_factory, image_root, detection_id)

    def handle_correct(detection_id: int, new_species: str) -> Optional[dict]:
        """Reassign a detection's species; returns the updated row or None."""
        return correct_detection_species(
            control_session_factory, image_root, detection_id, new_species
        )

    def list_custom() -> list[str]:
        """Return the user-added custom species labels."""
        return list_custom_species(control_session_factory)

    def register_custom(name: str) -> str:
        """Persist a new custom species label; returns its canonical form."""
        return add_custom_species(control_session_factory, name)

    deps = ControlServerDeps(
        crop_controller=camera.crop_controller,
        delete_detection=handle_delete,
        settings_controller=settings_controller,
        restart=_schedule_restart,
        correct_species=handle_correct,
        species_labels=species_labels,
        list_custom_species=list_custom,
        register_species=register_custom,
    )
    snapshot_port = camera_server_port()
    try:
        return start_camera_server(camera.picam2, snapshot_port, deps=deps)
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
    camera: Camera, manager: ClassificationManager, gating: Gating
) -> None:
    """Run the frame-capture loop until interrupted.

    Installs the per-frame detection callback (which updates the tracker and
    queues bird detections) and then repeatedly captures metadata and parses
    detections under the crop controller's lock.

    Args:
        camera: The started camera bundle.
        manager: The classification manager the callback feeds.
        gating: The stable-track gating bundle.
    """
    labels = get_labels(camera.intrinsics)
    # The IMX500 handles are constant across the loop; bundle them once.
    context = InferenceContext(camera.imx500, camera.intrinsics, camera.picam2)
    state: dict = {"last_results": None}

    def detection_callback(request):
        """Update tracker state and process detections for one frame."""
        gating.tracker.update_frame(state["last_results"] or [])
        process_detections(request, "main", state["last_results"], manager, labels)

    camera.picam2.pre_callback = detection_callback

    while True:
        # Hold the crop controller's lock across capture + parse so a UI-triggered
        # crop reconfigure (which may stop/start the camera) can never interleave
        # with an in-flight capture.
        with camera.crop_controller.camera_lock:
            metadata = camera.picam2.capture_metadata()
            results = parse_detections(
                metadata,
                context,
                app_config.threshold,
                # When the DNN input is restricted to the crop, the network's
                # boxes are ROI-relative; hand parse_detections the active ROI so
                # it remaps them back to full-sensor coords (None = full FOV).
                inference_roi=camera.inference_roi_state.roi,
            )
            # Keep only included object-detection classes (default just "bird")
            # before they reach the tracker or drawing, so non-bird false
            # positives never spawn tracks that flood the logs. Read live from
            # config so the Settings page can edit the include list without a
            # restart.
            state["last_results"] = filter_included_detections(
                results, labels, app_config.included_classes
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
    # Load the persisted settings overlay and push it onto the static config
    # *before* anything is wired, so restart-only settings shape the build.
    settings_path = settings_config_path()
    settings = load_settings(settings_path)
    apply_settings_to_config(settings)
    configure_logging(app_config.debug)

    # Create the database schema up front (the detector owns all DB writes).
    engine = make_engine()
    init_db(engine)

    # Rebuild the geomodel location prior if the configured location changed
    # (a no-op when unchanged, or when no location is set). Never blocks startup.
    refresh_geo_priors(
        make_session_factory(engine), app_config.latitude, app_config.longitude
    )

    # IMX500 must be initialised before Picamera2.
    imx500 = wait_for_camera(app_config.model)
    intrinsics = prepare_intrinsics(imx500)

    classifier = setup_classifier(
        str(classifier_model_path()), str(class_to_idx_path())
    )
    # The correction picker offers every real species (the "Unknown" non-bird
    # sentinel is not a valid re-identification target).
    species_labels = sorted(
        label
        for label in (classifier.idx_to_class or {}).values()
        if label != "Unknown"
    )
    gating = build_gating(intrinsics)

    # The detector owns all DB writes; the API mounts the same database read-only.
    detection_writer = DetectionWriter(make_session_factory(engine))
    # Build the geomodel Bayesian-update adjuster from the priors refreshed above
    # (None when no location is configured / no priors stored — classification
    # then runs unadjusted).
    geo_adjuster = build_geo_adjuster(classifier, make_session_factory(engine))
    manager = build_manager(classifier, gating, detection_writer, geo_adjuster)
    # The save-side classification settings live on the pipeline context.
    apply_settings_to_context(settings, manager.context)
    settings_controller = SettingsController(settings_path, settings, manager.context)

    camera = build_camera(imx500, intrinsics)
    # By default the clip records the cropped, ISP-processed `main` frame (so it
    # matches the saved still). Only when `video.full_fov` is enabled do we feed
    # the recorder the raw stream (debayered + downscaled) to record the whole,
    # uncropped field of view. Either way this is only wired when video recording
    # is on (the recorder is only built then).
    if gating.video_recorder is not None and app_config.video.full_fov:
        manager.context.video_frame_source = build_clip_frame_source(camera.picam2)
    control_server = _start_control_server(
        camera, engine, settings_controller, species_labels
    )

    try:
        _run_capture_loop(camera, manager, gating)
    except KeyboardInterrupt:
        _shutdown(control_server, manager, detection_writer)


if __name__ == "__main__":
    main()
