"""Camera bring-up: IMX500 init, network intrinsics, and the Picamera2 stream.

Everything needed to get from "no camera" to a started :class:`Camera` bundle
lives here so ``main.py`` can stay a short startup script.  Three steps, in
order:

* :func:`wait_for_camera` — initialise the IMX500 device, retrying until the
  camera dev-node appears (so a missing camera never crash-loops the detector).
* :func:`prepare_intrinsics` — apply the ``config.intrinsics`` overrides to the
  network intrinsics object.
* :func:`build_camera` — start ``Picamera2`` at the persisted crop and wire up
  the :class:`CropController` that owns subsequent live crop changes.

This module is Pi-only (it imports ``libcamera`` / ``picamera2``); nothing under
``ml/`` or the test suite imports it.
"""

import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import libcamera
import picamera2.formats as picamera2_formats  # type: ignore
from picamera2 import Picamera2  # type: ignore
from picamera2.devices import IMX500  # type: ignore
from picamera2.devices.imx500 import NetworkIntrinsics  # type: ignore
from picamera2.sensor_format import SensorFormat  # type: ignore

from birdscanner.detector.config.config import config as app_config
from birdscanner.detector.hardware.crop import (
    SENSOR_W,
    SENSOR_H,
    CropRegion,
    SensorDimensions,
    crop_config_path,
    default_crop_region,
    inference_roi_for_crop,
    load_crop_region,
    main_stream_size_for_crop,
)
from birdscanner.detector.hardware.crop_controller import (
    CropController,
    CropControllerConfig,
)
from birdscanner.detector.paths import coco_labels_path
from birdscanner.detector.hardware.raw_frame import FULL_FOV_RAW_SIZE
from birdscanner.ml.object_detection import InferenceRoi

logger = logging.getLogger("tracking")


def _full_fov_raw_stream(picam2: Picamera2) -> dict:
    """Return the ``raw`` stream spec for the full-FOV sensor mode.

    The video clip is fed from the ``raw`` stream (the only source of the
    uncropped field of view; see :mod:`birdscanner.detector.hardware.raw_frame`). The
    spec requests the *unpacked* format of the full-FOV binned mode so
    ``request.make_array("raw")`` yields a directly-demosaicable integer array
    (the native packed ``*_CSI2P`` format would need bespoke unpacking).

    The unpacked format is resolved via libcamera's lightweight
    ``generate_configuration`` enumeration (the non-allocating part of
    picamera2's ``sensor_modes``) rather than reading ``picam2.sensor_modes``
    directly: that property lazily calls ``configure()`` — allocating
    full-resolution DMA buffers — for *every* sensor mode, which exhausts the
    container's host-global CMA pool and crashes bring-up with
    ``OSError: [Errno 12] Cannot allocate memory`` before ``picam2.start()`` is
    even reached. Passing an explicit size AND format also means ``configure()``
    never needs ``sensor_modes`` at ``start()``. On any failure this falls back
    to a size-only spec (the clip pipeline then degrades gracefully to the
    cropped ``main`` frame — see :mod:`birdscanner.detector.hardware.raw_frame` — rather
    than crashing).

    Args:
        picam2: The ``Picamera2`` instance whose sensor formats are enumerated.

    Returns:
        A ``raw`` stream configuration dict for ``create_preview_configuration``.
    """
    try:
        raw_config = picam2.camera.generate_configuration([libcamera.StreamRole.Raw])
        raw_formats = raw_config.at(0).formats
        for pix in raw_formats.pixel_formats:
            name = str(pix)
            if not picamera2_formats.is_raw(name):
                continue
            sizes = {(size.width, size.height) for size in raw_formats.sizes(pix)}
            if FULL_FOV_RAW_SIZE in sizes:
                unpacked = SensorFormat(name).unpacked
                return {"size": FULL_FOV_RAW_SIZE, "format": str(unpacked)}
    except Exception:
        logger.exception(
            "Could not resolve full-FOV raw format via generate_configuration; "
            "falling back to size-only raw spec"
        )
    return {"size": FULL_FOV_RAW_SIZE}


@dataclass
class InferenceRoiState:
    """Mutable holder for the on-chip DNN inference ROI currently in force.

    Written by ``build_camera``'s ``apply_inference_roi`` (at boot and on every
    live crop change) and read by ``main._run_capture_loop`` so
    :func:`birdscanner.ml.object_detection.parse_detections` can remap the
    network's ROI-relative boxes back to full-sensor coordinates. It mirrors
    *exactly* the ROI pushed to ``IMX500.set_inference_roi_abs`` (a single source
    of truth), and stays ``None`` when inference is not restricted to the crop
    (``config.restrict_inference_to_crop`` off), which makes the remap a no-op.
    Both the writes and the read happen under the crop controller's
    ``camera_lock``, so no additional synchronisation is needed.

    Attributes:
        roi: The active inference ROI, or ``None`` when the DNN sees the full FOV.
    """

    roi: Optional[InferenceRoi] = None


@dataclass
class Camera:
    """The started camera and the objects needed to drive/reconfigure it.

    Attributes:
        picam2: The started ``Picamera2`` instance.
        imx500: The initialised IMX500 device.
        intrinsics: The (overridden) network intrinsics.
        crop_controller: Owns live crop changes and the shared camera lock.
        inference_roi_state: The on-chip DNN inference ROI currently in force,
            kept in sync with the crop for the capture loop's coordinate remap.
    """

    picam2: Any
    imx500: Any
    intrinsics: Any
    crop_controller: CropController
    inference_roi_state: InferenceRoiState


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


def prepare_intrinsics(imx500: IMX500) -> Any:
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


def build_camera(imx500: IMX500, intrinsics: Any) -> Camera:
    """Start the camera at the persisted crop and wrap it in a :class:`Camera`.

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
    # The full-FOV raw stream feeds the (uncropped) video clip, but only when the
    # full-FOV clip mode is enabled. By default the clip records the cropped
    # `main` frame (matching the still), so we skip the explicit raw request and
    # its unpacked-format CMA premium (~+14 MB). Computed once and reused across
    # reconfigures — it does not depend on the crop or main size.
    raw_stream = _full_fov_raw_stream(picam2) if app_config.video.full_fov else None

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
        # The raw stream is the full sensor field of view (ScalerCrop only affects
        # the processed `main` stream), used to record the uncropped video clip
        # when full-FOV mode is on; it is already allocated, so requesting it
        # explicitly only makes it accessible in the per-frame callback. Omitted
        # (``raw_stream is None``) in the default cropped-clip mode.
        raw_kwargs = {"raw": raw_stream} if raw_stream is not None else {}
        return picam2.create_preview_configuration(
            # picamera2's "888" format names are byte-reversed vs. the numpy
            # array they yield: "BGR888" delivers an [R, G, B]-ordered array. The
            # whole pipeline (ConvNeXt classifier, PIL thumbnails, the cv2
            # RGB2BGR writes) assumes RGB, so we must request BGR888 to actually
            # get RGB. Using "RGB888" here yields BGR and swaps red<->blue
            # everywhere downstream.
            main={"size": main_size, "format": "BGR888"},
            **raw_kwargs,
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

    sensor = SensorDimensions(SENSOR_W, SENSOR_H)
    inference_roi_state = InferenceRoiState()

    def apply_inference_roi(region: CropRegion) -> None:
        """Restrict the on-chip DNN input to the detection crop region.

        The IMX500 runs detection on-sensor, so ``ScalerCrop`` (which only crops
        the ISP ``main`` stream) does not change what the detector sees; without
        this the DNN squishes the whole field of view into its input tensor and a
        feeder bird scores poorly. Wired to the crop at boot and re-applied on
        every live crop change so the detector always sees the same zoomed view
        as the classifier. Gated by ``config.restrict_inference_to_crop`` so the
        historic full-FOV behaviour can be restored for comparison.

        Restricting the ROI makes the network return boxes normalized to the
        ROI, so the exact ROI pushed here is also published to
        ``inference_roi_state`` for the capture loop's coordinate remap (see
        :class:`InferenceRoiState`), keeping the two in lockstep.

        Args:
            region: The detection crop region to restrict inference to.
        """
        if not app_config.restrict_inference_to_crop:
            return
        roi = inference_roi_for_crop(region, sensor)
        imx500.set_inference_roi_abs(roi)
        inference_roi_state.roi = InferenceRoi(*roi, SENSOR_W, SENSOR_H)

    apply_inference_roi(crop_region)

    crop_controller = CropController(
        picam2,
        CropControllerConfig(
            region=crop_region,
            main_size=initial_main_size,
            config_factory=build_camera_config,
            config_path=crop_config_path(),
            sensor=sensor,
            set_inference_roi=apply_inference_roi,
        ),
    )
    return Camera(picam2, imx500, intrinsics, crop_controller, inference_roi_state)
