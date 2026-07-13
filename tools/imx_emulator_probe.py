#!/usr/bin/env python3
"""Compare the float YOLO11n ONNX vs the IMX500 int8 emulator over a clip.

Runs both detectors over every frame of a feeder clip and prints the diagnostics
that attribute the on-Pi low-confidence gap:

* ``top`` / ``mean_top10`` — the best and top-10-mean ``bird`` confidence.
* ``frames_hit`` — frames with at least one ``bird`` detection.
* ``bird_is_top`` — fraction of detected frames whose highest-confidence box is a
  ``bird`` (does quantization make the bird lose to another class?).
* ``best_area`` — the best ``bird`` box area as a fraction of the frame (size).

Three models are scored (all built / located under ``assets/models/``):

* ``float640`` — the generic float32 ``yolo11n.onnx`` (raw-tensor decode via
  :class:`OnnxYoloDetector`); the existing baseline.
* ``imx_float`` — the IMX-format *float* export (``yolo11n_imx_float.onnx``,
  boxes+scores decode via :class:`ImxOnnxDetector`); validates the IMX decoder
  against ``float640`` and is the clean baseline for isolating quantization.
* ``imx_int8`` — the Sony-MCT int8 fake-quant export
  (``yolo11n_imx_fakequant.onnx``); the emulator of the on-Pi ``.rpk``.

Read ``imx_int8`` against the live IMX500 reference (~0.27): if it collapses toward
it while ``imx_float`` stays ~0.8, int8 quantization is the cause; if it stays ~0.8,
quantization alone does not explain the gap. Build the IMX models with
``tools/build_imx_emulator.py`` (separate venv); until then only ``float640`` shows.

Runs on the project ``.venv`` (onnxruntime + cv2 only) — no torch/MCT needed here.
"""

import argparse
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Block disable so black re-wrapping can't drift the suppression off the anchor line.
# pylint: disable=wrong-import-position
from dev.emulation.yolo import OnnxYoloDetector, Detected, Detector
from dev.emulation.frames import VideoSource
from tools.imx_detector import ImxOnnxDetector

# pylint: enable=wrong-import-position

DEFAULT_CLIP = REPO_ROOT / "tests" / "_test_videos" / "great_tit_2.mp4"
FLOAT_MODEL = REPO_ROOT / "assets" / "models" / "yolo11n.onnx"
IMX_FLOAT_MODEL = REPO_ROOT / "assets" / "models" / "yolo11n_imx_float.onnx"
INT8_MODEL = REPO_ROOT / "assets" / "models" / "yolo11n_imx_fakequant.onnx"
TARGET_LABEL = "bird"
PI_REFERENCE_CONF = 0.27
# Low floor so genuinely-low (e.g. quantization-degraded) detections are not clipped.
PROBE_CONF = 0.01


class DetectorStats(NamedTuple):
    """Summary diagnostics for one detector over a clip."""

    name: str
    top: float
    mean_top10: float
    frames_hit: int
    bird_is_top_frac: float
    best_area_frac: float


def _load_frames(clip_path: Path) -> List[np.ndarray]:
    """Decode all RGB frames of a clip into memory.

    Args:
        clip_path: Path to the video clip.

    Returns:
        The decoded ``(H, W, 3)`` RGB frames.
    """
    source = VideoSource(str(clip_path), loop=False)
    frames: List[np.ndarray] = []
    while True:
        frame = source.next_frame()
        if frame is None:
            break
        frames.append(frame)
    source.release()
    return frames


def _bird_dets(dets: List[Detected]) -> List[Detected]:
    """Return only the ``bird`` detections."""
    return [d for d in dets if d.label == TARGET_LABEL]


def _box_area_frac(box) -> float:
    """Area of a normalized ``(x0, y0, x1, y1)`` box as a fraction of the frame."""
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


class _Scan(NamedTuple):
    """Raw per-frame accumulators from a detector pass over the clip."""

    bird_scores: List[float]  # best bird score per frame that had a bird
    best_area: float  # area of the single best-scoring bird box
    bird_is_top: int  # frames whose highest-confidence detection is a bird
    frames_with_any: int  # frames with >= 1 detection of any class


def _scan_frames(detector: Detector, frames: List[np.ndarray]) -> _Scan:
    """Run a detector over every frame and collect the raw accumulators.

    Args:
        detector: The detector to run.
        frames: The decoded RGB frames.

    Returns:
        The :class:`_Scan` accumulators (aggregated into stats by the caller).
    """
    bird_scores: List[float] = []
    best = (-1.0, float("nan"))  # (score, area) of the best bird box seen
    bird_is_top = 0
    frames_with_any = 0

    for frame in frames:
        dets = detector.detect(frame)
        if dets:
            frames_with_any += 1
            if max(dets, key=lambda d: d.score).label == TARGET_LABEL:
                bird_is_top += 1
        birds = _bird_dets(dets)
        if not birds:
            continue
        best_bird = max(birds, key=lambda d: d.score)
        bird_scores.append(best_bird.score)
        if best_bird.score > best[0]:
            best = (best_bird.score, _box_area_frac(best_bird.box))

    return _Scan(bird_scores, best[1], bird_is_top, frames_with_any)


def _run_detector(
    name: str, detector: Detector, frames: List[np.ndarray]
) -> DetectorStats:
    """Run a detector over every frame and aggregate the diagnostic stats.

    Args:
        name: Label for the detector (e.g. ``"float640"``).
        detector: The detector to run.
        frames: The decoded RGB frames.

    Returns:
        The aggregated :class:`DetectorStats`.
    """
    scan = _scan_frames(detector, frames)
    scores = np.array(scan.bird_scores, dtype=float)
    top = float(np.max(scores)) if scores.size else float("nan")
    mean_top10 = float(np.mean(np.sort(scores)[-10:])) if scores.size else float("nan")
    bird_is_top_frac = (
        scan.bird_is_top / scan.frames_with_any
        if scan.frames_with_any
        else float("nan")
    )
    return DetectorStats(
        name, top, mean_top10, scores.size, bird_is_top_frac, scan.best_area
    )


def _print_table(stats: List[DetectorStats]) -> None:
    """Print the comparison table + per-row deltas vs the baseline and Pi reference."""
    header = (
        f"{'detector':<12}{'top':>7}{'mean_top10':>12}{'frames_hit':>12}"
        f"{'bird_is_top':>13}{'best_area':>11}"
    )
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s.name:<12}{s.top:>7.2f}{s.mean_top10:>12.2f}{s.frames_hit:>12}"
            f"{s.bird_is_top_frac:>13.2f}{s.best_area_frac:>11.3f}"
        )
    print("-" * len(header))
    baseline = stats[0]
    for s in stats[1:]:
        print(
            f"{s.name} - {baseline.name}:  "
            f"d_top={s.top - baseline.top:+.2f}  "
            f"d_mean_top10={s.mean_top10 - baseline.mean_top10:+.2f}"
        )
    print(f"Pi live IMX500 reference (detection_confidence): ~{PI_REFERENCE_CONF:.2f}")
    print(
        "If the int8 row collapses toward ~0.27 (while imx_float stays ~0.8), int8 "
        "quantization is the cause; if int8 also stays ~0.8, quantization alone does not "
        "explain the gap."
    )


def _maybe_imx(
    name: str, model: Path, frames: List[np.ndarray]
) -> Optional[DetectorStats]:
    """Score an IMX-format model if it exists, else warn and return ``None``.

    Args:
        name: Row label (e.g. ``"imx_int8"``).
        model: Path to the IMX-format ``.onnx``.
        frames: The decoded RGB frames.

    Returns:
        The :class:`DetectorStats`, or ``None`` when the model file is absent.
    """
    if not model.exists():
        print(
            f"({name} model not found: {model.name} — run tools/build_imx_emulator.py)"
        )
        return None
    detector = ImxOnnxDetector(str(model), conf_threshold=PROBE_CONF)
    return _run_detector(name, detector, frames)


def probe(
    clip_path: Path, float_model: Path, imx_float_model: Path, int8_model: Path
) -> None:
    """Score the generic-float, IMX-float and IMX-int8 models over the clip and report.

    Args:
        clip_path: Clip to score.
        float_model: Generic float32 ``yolo11n.onnx`` (raw-tensor decode) — baseline.
        imx_float_model: IMX-format float ONNX (decoder validation / clean baseline).
        int8_model: IMX-format int8 fake-quant ONNX (the emulator).
    """
    frames = _load_frames(clip_path)
    print(
        f"clip: {clip_path.name} — {len(frames)} frames "
        f"({frames[0].shape[1]}x{frames[0].shape[0]})\n"
    )

    stats: List[DetectorStats] = [
        _run_detector(
            "float640",
            OnnxYoloDetector(str(float_model), conf_threshold=PROBE_CONF),
            frames,
        )
    ]
    for name, model in (("imx_float", imx_float_model), ("imx_int8", int8_model)):
        row = _maybe_imx(name, model, frames)
        if row is not None:
            stats.append(row)
    _print_table(stats)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip", type=Path, default=DEFAULT_CLIP, help="Video clip to score"
    )
    parser.add_argument(
        "--float-model",
        type=Path,
        default=FLOAT_MODEL,
        help="generic float32 yolo11n.onnx",
    )
    parser.add_argument(
        "--imx-float-model",
        type=Path,
        default=IMX_FLOAT_MODEL,
        help="IMX-format float onnx (decoder-validation baseline)",
    )
    parser.add_argument(
        "--int8-model",
        type=Path,
        default=INT8_MODEL,
        help="IMX-format int8 fake-quant onnx",
    )
    args = parser.parse_args()
    probe(args.clip, args.float_model, args.imx_float_model, args.int8_model)


if __name__ == "__main__":
    main()
