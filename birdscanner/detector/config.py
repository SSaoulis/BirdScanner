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
    """

    save: bool = True
    pre_roll_seconds: float = 3.0
    post_roll_seconds: float = 4.0


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
        debug: Enable DEBUG-level logging for track lifecycle events.
        preview: Show the camera preview window.
        latitude: Camera location latitude in degrees, ``[-90, 90]``. Seeds the
            geolocation prior cache; changing it (via Settings) regenerates the
            per-week presence vectors on the next boot.
        longitude: Camera location longitude in degrees, ``[-180, 180]`` (see
            ``latitude``).
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
    debug: bool = True
    preview: bool = False
    latitude: float = 0.0
    longitude: float = 0.0
    intrinsics: IntrinsicsConfig = field(default_factory=IntrinsicsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


config = Config()
