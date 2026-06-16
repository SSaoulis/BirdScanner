"""Application configuration.

Replaces the former command-line arguments. Edit the values on the module-level
``config`` instance below to change runtime behaviour; ``main.py`` reads from it
instead of parsing ``argparse`` flags.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Runtime configuration for the bird detection application.

    Attributes:
        model: Path to the IMX500 object-detection network (``.rpk``) firmware.
        fps: Camera frame rate; ``None`` falls back to the network intrinsics rate.
        bbox_normalization: Whether the model emits normalized bbox coordinates;
            ``None`` leaves the network intrinsics default untouched.
        bbox_order: Bounding-box coordinate order — ``"yx"`` -> (y0, x0, y1, x1)
            or ``"xy"`` -> (x0, y0, x1, y1).
        threshold: Minimum detection confidence required to keep a detection.
        ignore_dash_labels: Whether to drop ``'-'`` labels; ``None`` leaves the
            network intrinsics default untouched.
        preserve_aspect_ratio: Whether to preserve the input tensor aspect ratio;
            ``None`` leaves the network intrinsics default untouched.
        labels: Path to a labels file overriding the network intrinsics labels;
            ``None`` uses the bundled defaults.
        print_intrinsics: When ``True``, print the network intrinsics and exit.
        multithread: Run classification on a background thread so the camera
            callback never blocks.
        object_duration_threshold: Seconds a track must be stable (IoU>0.6 across
            frames) before bird classification fires; ``0`` reverts to the legacy
            per-frame logic.
        debug: Enable DEBUG-level logging for track lifecycle events.
        preview: Show the camera preview window.
    """

    model: str = (
        "/usr/share/imx500-models/"
        "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    )
    fps: Optional[int] = None
    bbox_normalization: Optional[bool] = None
    bbox_order: str = "yx"
    threshold: float = 0.55
    ignore_dash_labels: Optional[bool] = None
    preserve_aspect_ratio: Optional[bool] = None
    labels: Optional[str] = None
    print_intrinsics: bool = False
    multithread: bool = True
    object_duration_threshold: float = 0.2
    debug: bool = True
    preview: bool = False


config = Config()
