"""Application configuration.

Replaces the former command-line arguments. Edit the values on the module-level
``config`` instance below to change runtime behaviour; ``main.py`` reads from it
instead of parsing ``argparse`` flags.

Settings are grouped into cohesive sub-configs so the top-level :class:`Config`
stays small: :class:`IntrinsicsConfig` holds the network-intrinsics overrides
(the values ``main.py`` pushes onto the IMX500 intrinsics object) and
:class:`VideoConfig` holds the per-detection mp4-clip settings.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IntrinsicsConfig:
    """Network-intrinsics overrides applied to the IMX500 intrinsics object.

    ``main.py`` iterates ``vars(config.intrinsics)`` and, for every field the
    network intrinsics object exposes, overrides it (skipping ``None`` so the
    network defaults are left untouched). ``labels`` is special-cased: it is a
    path to a labels file that is read and applied.

    Attributes:
        fps: Camera frame rate; ``None`` falls back to the network intrinsics rate.
        bbox_normalization: Whether the model emits normalized bbox coordinates;
            ``None`` leaves the network intrinsics default untouched.
        bbox_order: Bounding-box coordinate order — ``"yx"`` -> (y0, x0, y1, x1)
            or ``"xy"`` -> (x0, y0, x1, y1).
        ignore_dash_labels: Whether to drop ``'-'`` labels; ``None`` leaves the
            network intrinsics default untouched.
        preserve_aspect_ratio: Whether to preserve the input tensor aspect ratio;
            ``None`` leaves the network intrinsics default untouched.
        labels: Path to a labels file overriding the network intrinsics labels;
            ``None`` uses the bundled defaults.
    """

    fps: Optional[int] = None
    bbox_normalization: Optional[bool] = True
    bbox_order: str = "xy"
    ignore_dash_labels: Optional[bool] = None
    preserve_aspect_ratio: Optional[bool] = None
    labels: Optional[str] = None


@dataclass
class VideoConfig:
    """Per-detection mp4-clip settings.

    Attributes:
        save: Save a short mp4 clip around each saved detection (in addition to
            the still image); disabling reverts to image-only saving.
        pre_roll_seconds: Seconds of buffered footage to prepend to a clip.
        post_roll_seconds: Seconds of footage to keep recording after the
            detection triggers a clip.
        full_fov: Record the whole, uncropped field of view from the camera's
            raw sensor stream. ``False`` (the default) records the cropped,
            ISP-processed ``main`` stream instead — the clip then matches the
            saved still exactly (same crop, same image quality). ``True`` reverts
            to the full-scene clip, which is debayered from the raw stream with no
            ISP processing, so it is softer/noisier and 4:3 (see
            :mod:`birdscanner.detector.hardware.raw_frame`); it also costs extra CMA to
            request the raw stream in an unpacked format (see
            :func:`birdscanner.detector.hardware.camera.build_camera`).
    """

    save: bool = True
    pre_roll_seconds: float = 3.0
    post_roll_seconds: float = 4.0
    full_fov: bool = False


@dataclass
class Config:
    """Runtime configuration for the bird detection application.

    Attributes:
        model: Path to the IMX500 object-detection network (``.rpk``) firmware.
        threshold: Minimum detection confidence required to keep a detection.
        print_intrinsics: When ``True``, print the network intrinsics and exit.
        multithread: Run classification on a background thread so the camera
            callback never blocks.
        object_duration_threshold: Seconds a track must be stable (IoU>0.6 across
            frames) before bird classification fires; values ``<= 0`` floor the
            requirement to a single stable frame.
        restrict_inference_to_crop: Crop the on-chip detector's input to the
            detection crop region (via ``IMX500.set_inference_roi_abs``) so the
            DNN sees the same zoomed-in feeder view as the classifier instead of
            the whole sensor field of view. ``True`` (the default) fixes the
            small-bird low-confidence problem; ``False`` reverts to the historic
            full-FOV inference for comparison. Applied at boot and kept in sync
            on every live crop change (see
            :func:`birdscanner.detector.hardware.crop.inference_roi_for_crop`).
        included_classes: Object-detection (YOLO/COCO) class labels to keep
            before tracking, matched case-insensitively (an allowlist, default
            ``{"bird"}``). The IMX500 model emits every COCO class it sees, so
            anything not on this list (e.g. ``"bench"``, ``"person"``) is dropped
            in the capture loop before it can create tracks that flood the logs.
            An empty set keeps everything. Applied live.
        debug: Enable DEBUG-level logging for track lifecycle events.
        preview: Show the camera preview window.
        latitude: Deployment latitude in degrees, used to compute the geomodel
            spatio-temporal prior; ``None`` until a location is configured, in
            which case the prior is not built.
        longitude: Deployment longitude in degrees (see ``latitude``).
        intrinsics: Network-intrinsics overrides (see :class:`IntrinsicsConfig`).
        video: Per-detection clip settings (see :class:`VideoConfig`).
    """

    # These defaults mirror the known-good runtime invocation:
    #   main.py --model .../imx500_network_yolo11n_pp.rpk \
    #           --bbox-normalization --bbox-order xy \
    #           --object-duration-threshold 0.1 --multithread --debug
    # The detector uses the YOLO11n network, whose post-processed output tensor
    # emits normalized, xy-ordered boxes. Reverting to the SSD model / yx order /
    # un-normalized boxes makes parse_detections misdecode the tensor and emit a
    # single garbage full-frame detection.
    model: str = "/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk"
    threshold: float = 0.55
    print_intrinsics: bool = False
    multithread: bool = True
    object_duration_threshold: float = 0.1
    restrict_inference_to_crop: bool = True
    included_classes: set[str] = field(default_factory=lambda: {"bird"})
    debug: bool = True
    preview: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    intrinsics: IntrinsicsConfig = field(default_factory=IntrinsicsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


config = Config()
