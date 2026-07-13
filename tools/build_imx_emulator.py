#!/usr/bin/env python3
"""Build a host-runnable *int8 emulator* of the IMX500 YOLO11n ``.rpk``.

The live IMX500 on-chip detector scores far lower than our float ONNX on the same
scene (see ``notebooks/tracking_playground.ipynb``). Orientation and colour were
ruled out; the remaining suspect is the **int8 quantization** the IMX500 pipeline
applies (our ``yolo11n.onnx`` is float32). This script reproduces that quantization
on the host with Sony's Model Compression Toolkit (MCT) — the same quantizer the
IMX500 toolchain uses — and exports a **fake-quant ONNX** (values on the int8 grid,
stored as float32) that runs in plain ``onnxruntime``.

The exported model keeps YOLO11n's raw ``(1, 84, 8400)`` output (no NMS wrapper),
so it is a drop-in for :class:`dev.emulation.yolo.OnnxYoloDetector` — letterbox,
decode and NMS are then identical to the float run and the *only* difference is
int8 weights/activations. Compare the two with ``tools/imx_emulator_probe.py``.

Fidelity: this emulates the IMX500 *quantization scheme* on a generic
``yolo11n.pt`` calibrated on the feeder clip — it is not a bit-exact clone of
Raspberry Pi's packaged ``.rpk`` (different calibration set; no RPi converter
step). It answers "can IMX-style int8 quantization drop confidence toward the
observed ~0.27?", not "reproduce the ``.rpk`` exactly".

Run in a SEPARATE venv (heavy torch/MCT deps — see ``requirements.imx-emulator.txt``):

    py -m venv .venv-imx
    .venv-imx\\Scripts\\pip install -r requirements.imx-emulator.txt
    .venv-imx\\Scripts\\python tools/build_imx_emulator.py

Writes ``assets/models/yolo11n_imx_fakequant.onnx`` (git-ignored, out-of-band).
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List

import numpy as np

# Anchor to the repo root so ``dev.emulation`` and the asset paths resolve when
# this is run from anywhere (mirrors the notebook's sys.path insert).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact letterbox the off-Pi detector uses, so calibration frames are
# preprocessed identically to inference (no train/serve skew). Block disable so
# black re-wrapping can't drift the suppression off the anchor line.
# pylint: disable=wrong-import-position
from dev.emulation.yolo import _letterbox
from dev.emulation.frames import VideoSource

# pylint: enable=wrong-import-position

DEFAULT_CLIPS = [
    REPO_ROOT / "tests" / "_test_videos" / "great_tit_2.mp4",
    REPO_ROOT / "tests" / "_test_videos" / "great_tit_1.mp4",
]
DEFAULT_PT = "yolo11n.pt"
DEFAULT_OUT = REPO_ROOT / "assets" / "models" / "yolo11n_imx_fakequant.onnx"
DEFAULT_IMGSZ = 640
DEFAULT_NUM_CALIB = 200
DEFAULT_TPC_VERSION = "1.0"


def _preprocess(frame: np.ndarray, imgsz: int) -> np.ndarray:
    """Letterbox an RGB frame to ``imgsz`` and return a ``(3, imgsz, imgsz)`` float32 tensor.

    Matches :meth:`dev.emulation.yolo.OnnxYoloDetector.detect` preprocessing
    (letterbox → ``/255`` → CHW, RGB), so calibration statistics reflect the real
    inference distribution.

    Args:
        frame: Source ``(H, W, 3)`` uint8 RGB frame.
        imgsz: Square network input side length.

    Returns:
        The preprocessed ``(3, imgsz, imgsz)`` float32 CHW tensor in ``[0, 1]``.
    """
    canvas, _ = _letterbox(frame, imgsz)
    return np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))


def _collect_calibration_frames(
    clip_paths: List[Path], imgsz: int, limit: int
) -> List[np.ndarray]:
    """Preprocess up to ``limit`` frames sampled across the given clips.

    Args:
        clip_paths: Video clips to draw calibration frames from.
        imgsz: Network input side length.
        limit: Maximum number of preprocessed frames to return.

    Returns:
        A list of ``(3, imgsz, imgsz)`` float32 tensors (may be shorter than
        ``limit`` if the clips are short).

    Raises:
        FileNotFoundError: If none of the clips exist / decode.
    """
    tensors: List[np.ndarray] = []
    for path in clip_paths:
        if not path.exists():
            print(f"  (skipping missing clip: {path})")
            continue
        source = VideoSource(str(path), loop=False)
        while len(tensors) < limit:
            frame = source.next_frame()
            if frame is None:
                break
            tensors.append(_preprocess(frame, imgsz))
        source.release()
        if len(tensors) >= limit:
            break
    if not tensors:
        raise FileNotFoundError(
            f"No calibration frames decoded from any of: {clip_paths}"
        )
    print(f"  collected {len(tensors)} calibration frames")
    return tensors


def _make_representative_data_gen(
    tensors: List[np.ndarray],
) -> Callable[[], Iterator[List[np.ndarray]]]:
    """Return an MCT ``representative_data_gen`` yielding one-sample batches.

    MCT calls the returned callable to obtain an iterator; each yielded item is a
    list of model inputs (here a single ``(1, 3, H, W)`` float32 batch).

    Args:
        tensors: Preprocessed ``(3, H, W)`` calibration tensors.

    Returns:
        A zero-arg callable producing a fresh iterator over the calibration set.
    """

    def representative_data_gen() -> Iterator[List[np.ndarray]]:
        for tensor in tensors:
            yield [tensor[np.newaxis, ...]]

    return representative_data_gen


def _load_export_model(pt_path: str, imgsz: int):
    """Load YOLO11n wrapped for ``torch.fx`` tracing (the form MCT quantizes).

    A raw Ultralytics YOLO module is not ``torch.fx``-traceable: the ``C2f`` /
    ``C3k2`` blocks use ``list(x.chunk(2, 1))`` (an unsupported ``len`` on a proxy)
    and the ``Detect`` head has non-traceable post-processing. This mirrors exactly
    what Ultralytics' own ``format=imx`` export does to make it work:

    * swap every ``C2f`` block's ``forward`` to ``forward_split`` (uses ``.split()``
      instead of ``chunk`` — fx-traceable), and
    * wrap the model in Ultralytics' :class:`FXModel`, which rebinds a traceable
      ``_inference`` on the ``Detect`` head.

    The resulting model's ``Detect`` head emits the IMX ``_pp`` format — a
    ``(1, N, 4)`` boxes tensor (``cx, cy, w, h`` in input-pixel space) and a
    ``(1, N, 80)`` sigmoid class-scores tensor — decoded by
    :class:`tools.imx_detector.ImxOnnxDetector` (this is also what the real
    ``yolo11n_pp.rpk`` produces, so it is a *closer* match than the raw tensor).

    Args:
        pt_path: Path/name of the YOLO11n ``.pt`` (auto-downloaded by ultralytics).
        imgsz: Network input side length (anchors are built for this size).

    Returns:
        An fx-traceable ``nn.Module`` in eval mode emitting ``(boxes, scores, ...)``.
    """
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f
    from ultralytics.utils.export.imx import FXModel

    model = YOLO(pt_path).model
    model.eval()
    for module in model.modules():
        if isinstance(module, C2f):
            module.forward = module.forward_split
    return FXModel(model, imgsz=(imgsz, imgsz)).eval()


def _get_imx500_tpc(tpc_version: str):
    """Return the IMX500 target-platform capabilities for the given TPC version.

    Uses the MCT 2.x signature ``get_target_platform_capabilities(tpc_version,
    device_type)``. The IMX500 TPC versions available in MCT 2.6 are ``1.0``,
    ``4.0``, ``5.0``, ``6.0``; the default here is ``1.0`` (the baseline 8-bit
    IMX500 profile — enough to expose int8 quantization behaviour).

    Args:
        tpc_version: The IMX500 TPC version string (e.g. ``"1.0"``).

    Returns:
        An MCT target-platform-capabilities object for the IMX500.
    """
    import model_compression_toolkit as mct

    return mct.get_target_platform_capabilities(tpc_version, "imx500")


@dataclass
class BuildOptions:
    """Bundled parameters for :func:`build` (keeps its signature under the arg limit).

    Attributes:
        pt_path: YOLO11n ``.pt`` path/name (auto-downloaded by ultralytics).
        clip_paths: Clips to calibrate on.
        out_path: Where to write the fake-quant ``.onnx``.
        imgsz: Network input side length (640 for the IMX500 YOLO11n).
        num_calib: Number of calibration frames.
        tpc_version: IMX500 TPC version passed to MCT.
    """

    pt_path: str
    clip_paths: List[Path]
    out_path: Path
    imgsz: int
    num_calib: int
    tpc_version: str


def _export_fakequant_onnx(mct, quant_model, out_path: Path, repr_dataset) -> None:
    """Export the MCT fake-quant model to ONNX, forcing torch's legacy exporter.

    MCT 2.6's ``FAKELY_QUANT`` exporter calls ``torch.onnx.export`` without
    ``dynamo=False``; on torch >= 2.9 that defaults to the new dynamo /
    ``torch.export`` path, which fails on this graph (``Detected mismatch between
    the structure of inputs and dynamic_shapes``). Temporarily force the legacy
    TorchScript exporter — the same one the float baseline export uses cleanly.

    Args:
        mct: The imported ``model_compression_toolkit`` module.
        quant_model: The MCT-quantized model to export.
        out_path: Destination ``.onnx`` path.
        repr_dataset: The representative-data generator MCT re-runs during export.
    """
    import torch

    original_export = torch.onnx.export

    def _legacy_export(*args, **kwargs):
        """Call the original exporter with the legacy TorchScript path forced."""
        kwargs.setdefault("dynamo", False)
        return original_export(*args, **kwargs)

    torch.onnx.export = _legacy_export
    try:
        mct.exporter.pytorch_export_model(
            model=quant_model,
            save_model_path=str(out_path),
            repr_dataset=repr_dataset,
            quantization_format=mct.exporter.QuantizationFormat.FAKELY_QUANT,
        )
    finally:
        torch.onnx.export = original_export


def build(
    options: "BuildOptions",
) -> None:
    """Quantize YOLO11n with MCT and export a fake-quant ONNX emulator.

    Args:
        options: The bundled build parameters (see :class:`BuildOptions`).
    """
    import torch
    import model_compression_toolkit as mct

    print("1/5 collecting calibration frames...")
    tensors = _collect_calibration_frames(
        options.clip_paths, options.imgsz, options.num_calib
    )
    representative_data_gen = _make_representative_data_gen(tensors)

    print("2/5 loading yolo11n (fx-traceable IMX export form)...")
    model = _load_export_model(options.pt_path, options.imgsz)

    # Also export the *float* (un-quantized) FXModel to ONNX. Decoded by the same
    # ImxOnnxDetector, it is the apples-to-apples baseline that isolates the int8
    # effect (imx_float vs imx_int8) and validates the decoder against the generic
    # float640 run. Best-effort: a failure here must not block the essential int8
    # export, so it is caught and logged (the probe simply omits the imx_float row).
    float_path = options.out_path.with_name(
        options.out_path.stem.replace("_fakequant", "") + "_float.onnx"
    )
    print(f"3/5 exporting float baseline ONNX -> {float_path}")
    options.out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.from_numpy(tensors[0][np.newaxis, ...])
    try:
        # dynamo=False forces the legacy TorchScript exporter (well-tested on YOLO,
        # and avoids the newer onnxscript-based path when it is unavailable).
        torch.onnx.export(model, dummy, str(float_path), opset_version=20, dynamo=False)
    except Exception as exc:
        print(f"    WARNING: float baseline export failed ({exc}); continuing to int8")

    print(
        f"4/5 running MCT post-training quantization (IMX500 TPC v{options.tpc_version})..."
    )
    tpc = _get_imx500_tpc(options.tpc_version)
    quant_model, _ = mct.ptq.pytorch_post_training_quantization(
        model,
        representative_data_gen,
        target_platform_capabilities=tpc,
    )

    print(f"5/5 exporting fake-quant ONNX -> {options.out_path}")
    _export_fakequant_onnx(mct, quant_model, options.out_path, representative_data_gen)
    _sanity_check(options.out_path, tensors[0], options.imgsz)
    _sanity_check(float_path, tensors[0], options.imgsz)


def _sanity_check(out_path: Path, sample: np.ndarray, imgsz: int) -> None:
    """Load the exported ONNX in onnxruntime and confirm the IMX output format.

    Confirms the model runs on the host and emits the ``(1, N, 4)`` boxes +
    ``(1, N, 80)`` scores tensors :class:`tools.imx_detector.ImxOnnxDetector`
    decodes, before the probe relies on it.

    Args:
        out_path: The exported ONNX path.
        sample: One preprocessed ``(3, H, W)`` calibration tensor to run.
        imgsz: Network input side length (for the log line).
    """
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(out_path), providers=["CPUExecutionProvider"]
        )
        name = session.get_inputs()[0].name
        outputs = session.run(None, {name: sample[np.newaxis, ...].astype(np.float32)})
        shapes = [np.asarray(o).shape for o in outputs]
        has_boxes = any(s[-1] == 4 for s in shapes if len(s) == 3)
        has_scores = any(s[-1] == 80 for s in shapes if len(s) == 3)
        print(
            f"    OK: onnxruntime loads {out_path.name}; input {imgsz}x{imgsz}, "
            f"outputs {shapes}"
        )
        if not (has_boxes and has_scores):
            print(
                "    WARNING: expected a (1, N, 4) boxes and (1, N, 80) scores output; "
                "inspect the export if ImxOnnxDetector misreads it."
            )
    except Exception as exc:  # pragma: no cover
        print(f"    WARNING: could not sanity-load {out_path.name}: {exc}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pt", default=DEFAULT_PT, help="YOLO11n .pt (auto-downloaded)"
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        type=Path,
        default=DEFAULT_CLIPS,
        help="Video clips to calibrate on",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output .onnx path"
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ, help="Network input size"
    )
    parser.add_argument(
        "--num-calib",
        type=int,
        default=DEFAULT_NUM_CALIB,
        help="Calibration frame count",
    )
    parser.add_argument(
        "--tpc-version",
        default=DEFAULT_TPC_VERSION,
        help="IMX500 TPC version for MCT (e.g. 1.0, 4.0, 5.0, 6.0)",
    )
    args = parser.parse_args()
    build(
        BuildOptions(
            pt_path=args.pt,
            clip_paths=args.clips,
            out_path=args.out,
            imgsz=args.imgsz,
            num_calib=args.num_calib,
            tpc_version=args.tpc_version,
        )
    )


if __name__ == "__main__":
    main()
