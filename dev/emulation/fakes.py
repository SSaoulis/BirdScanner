"""Fake ``picamera2`` / ``libcamera`` objects for the off-Pi camera emulator.

These stand-ins implement exactly the surface the real detector uses (enumerated
in the module docstrings of :mod:`birdscanner.detector.camera` and
:mod:`birdscanner.ml.object_detection`), so the *real* camera bring-up and the
*real* capture loop run unchanged against them off the Pi.

The wiring is deliberately indirect: the real ``build_camera`` constructs
``Picamera2(camera_num)`` with no reference to the IMX500, so a single emulation
state (frame source + detector + the shared fake IMX500) lives in the
module-level :data:`_STATE`.  :func:`set_emulation_state` installs it; the fakes
read from it.  :mod:`dev.emulation.install` registers these
objects as the ``picamera2`` / ``libcamera`` modules in ``sys.modules``.

Nothing Pi-only is imported here — these objects *are* the stand-ins.
"""

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from birdscanner.ml.object_detection import get_labels

from dev.emulation.frames import FrameSource
from dev.emulation.yolo import Detected, Detector

logger = logging.getLogger("tracking")

# Square network input the emulated IMX500 reports. Only its ratio matters: the
# real ``_decode_boxes`` divides boxes by this height and the fake multiplies by
# it, so the value cancels — any consistent size works.
_INPUT_SIZE: Tuple[int, int] = (640, 640)


class FakeNetworkIntrinsics:
    """Stand-in for ``picamera2`` ``NetworkIntrinsics``.

    Carries just the attributes ``prepare_intrinsics`` / ``build_gating`` read or
    write.  ``labels`` starts ``None`` so ``prepare_intrinsics`` loads the bundled
    COCO labels, exactly as it does on the Pi.
    """

    def __init__(self) -> None:
        """Initialise object-detection intrinsics with sensible defaults."""
        self.task = "object detection"
        self.labels: Optional[List[str]] = None
        self.inference_rate = 15
        self.preserve_aspect_ratio = False
        self.bbox_normalization = True
        self.bbox_order = "xy"
        self.ignore_dash_labels: Optional[bool] = None

    def update_with_defaults(self) -> None:
        """No-op: the fake already carries complete defaults."""


class FakeIMX500:
    """Stand-in for ``picamera2.devices.IMX500``.

    Holds the detections for the current frame (set each tick by
    :class:`FakePicam2`) and turns them into an inference tensor in the real
    ``.rpk`` convention, so the production ``parse_detections`` / ``_decode_boxes``
    run faithfully against it.
    """

    def __init__(self) -> None:
        """Initialise with empty detections and fresh network intrinsics."""
        self.network_intrinsics = FakeNetworkIntrinsics()
        self.camera_num = 0
        self._input_size = _INPUT_SIZE
        self.current_detections: List[Detected] = []

    # -- bring-up no-ops -------------------------------------------------------
    def show_network_fw_progress_bar(self) -> None:
        """No-op: there is no firmware upload to track off-Pi."""

    def set_auto_aspect_ratio(self) -> None:
        """No-op: aspect handling is irrelevant for the emulator."""

    # -- inference surface -----------------------------------------------------
    def get_input_size(self) -> Tuple[int, int]:
        """Return the ``(width, height)`` network input size."""
        return self._input_size

    def _tensor_rows(self) -> Tuple[List[List[float]], List[float], List[float]]:
        """Build the (boxes, scores, classes) rows for the current detections.

        Boxes are xy-ordered xyxy scaled to ``input_h`` pixels (the ``.rpk``
        convention); classes are resolved by **name** against the filtered
        intrinsics labels, dropping any detection whose class name is absent.
        """
        labels = get_labels(self.network_intrinsics)
        index_by_name = {name.lower(): idx for idx, name in enumerate(labels)}
        input_h = self._input_size[1]

        boxes: List[List[float]] = []
        scores: List[float] = []
        classes: List[float] = []
        for det in self.current_detections:
            class_index = index_by_name.get(det.label.lower())
            if class_index is None:
                continue
            x0, y0, x1, y1 = det.box
            boxes.append([x0 * input_h, y0 * input_h, x1 * input_h, y1 * input_h])
            scores.append(det.score)
            classes.append(float(class_index))
        return boxes, scores, classes

    def get_outputs(
        self, metadata: dict, add_batch: bool = True
    ) -> Optional[List[np.ndarray]]:
        """Return the current frame's detections as an IMX500-style tensor.

        The output mirrors the real post-processed ``.rpk`` tensor
        ``[boxes(1, N, 4), scores(1, N), classes(1, N)]`` with **xy-ordered**
        boxes in ``input_h``-pixel units, so the production ``_decode_boxes``
        normalization (``/ input_h``) and ``[1, 0, 3, 2]`` reorder reproduce the
        expected ``(y0, x0, y1, x1)`` per-detection tuple.

        Args:
            metadata: Ignored (the fake serves the single-threaded current
                frame); present to match the real signature.
            add_batch: Ignored; the fake always emits the batch dimension.

        Returns:
            The three-tensor list (an empty frame yields zero-length tensors,
            matching "no detections").
        """
        del metadata, add_batch
        boxes, scores, classes = self._tensor_rows()
        count = len(boxes)
        return [
            np.array(boxes, dtype=np.float32).reshape((1, count, 4)),
            np.array(scores, dtype=np.float32).reshape((1, count)),
            np.array(classes, dtype=np.float32).reshape((1, count)),
        ]

    def convert_inference_coords(
        self, box: Tuple[float, float, float, float], metadata: dict, picam2: Any
    ) -> Tuple[int, int, int, int]:
        """Map a normalized ``(y0, x0, y1, x1)`` box to ``main``-stream pixels.

        Mirrors the real ``convert_inference_coords`` closely enough for the
        pipeline: the box is scaled to the fake camera's ``main`` stream size so
        it aligns with the frame ``process_detections`` reads back.

        Args:
            box: The decoded ``(y0, x0, y1, x1)`` box in ``[0, 1]`` fractions.
            metadata: Ignored (no sensor transform to undo off-Pi).
            picam2: The :class:`FakePicam2` whose ``main`` size the box maps into.

        Returns:
            The box as ``(x, y, w, h)`` in ``main``-stream pixel coordinates.
        """
        del metadata
        # ``_decode_boxes`` yields per-axis numpy scalars; coerce to plain floats.
        y0, x0, y1, x1 = (float(v) for v in box)
        width, height = getattr(picam2, "main_size", None) or self._input_size
        x = int(round(x0 * width))
        y = int(round(y0 * height))
        box_w = int(round((x1 - x0) * width))
        box_h = int(round((y1 - y0) * height))
        return (max(0, x), max(0, y), max(0, box_w), max(0, box_h))


class FakeRequest:
    """Stand-in for a ``picamera2`` capture request.

    Wraps a single RGB ``main``-stream frame and a ScalerCrop tuple.
    """

    def __init__(self, frame: np.ndarray, scaler: Tuple[int, int, int, int]) -> None:
        """Record the frame and the ScalerCrop the request reports."""
        self._frame = frame
        self._scaler = scaler
        self.released = False

    def make_array(self, stream: str) -> np.ndarray:
        """Return the frame for any requested stream (``main``/``raw``)."""
        del stream
        return self._frame

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata carrying the current ScalerCrop."""
        return {"ScalerCrop": self._scaler}

    def release(self) -> None:
        """Mark the request released (no buffers to free off-Pi)."""
        self.released = True


class FakeMappedArray:
    """Stand-in for ``picamera2.MappedArray``.

    A context manager exposing the request's frame as a writable ``.array`` (so
    ``process_detections`` can draw boxes back onto it).
    """

    def __init__(self, request: FakeRequest, stream: str) -> None:
        """Record the request + stream to map on ``__enter__``."""
        self._request = request
        self._stream = stream
        self.array: Optional[np.ndarray] = None

    def __enter__(self) -> "FakeMappedArray":
        """Expose the request's frame as the writable ``.array``."""
        self.array = self._request.make_array(self._stream)
        return self

    def __exit__(self, *exc: Any) -> None:
        """Leave the frame in place; do not suppress exceptions."""


class FakePicam2:
    """Stand-in for ``picamera2.Picamera2`` that drives the emulated camera.

    Its ``capture_metadata`` is the frame pump: each call pulls the next frame
    from the emulation's :class:`FrameSource`, resizes it to the configured
    ``main`` size, runs the detector, invokes ``pre_callback`` with a fake
    request, and returns metadata — reproducing the per-frame contract the real
    picamera2 gives the capture loop.
    """

    def __init__(self, camera_num: int = 0) -> None:
        """Initialise an unstarted fake camera bound to the emulation state.

        Args:
            camera_num: Accepted for signature parity; unused.

        Raises:
            RuntimeError: If no emulation state has been installed.
        """
        del camera_num
        self._state = _require_state()
        self.main_size: Tuple[int, int] = _INPUT_SIZE
        self._scaler: Tuple[int, int, int, int] = (0, 0, *_INPUT_SIZE)
        self.pre_callback: Any = None
        self.started = False
        self._frame_count = 0
        self._last_frame: Optional[np.ndarray] = None
        # ``camera`` is only touched by the full-FOV raw path (disabled by
        # default); a minimal stub keeps that branch from crashing if reached.
        self.camera = SimpleNamespace(
            generate_configuration=lambda roles: SimpleNamespace(
                at=lambda _index: SimpleNamespace(
                    formats=SimpleNamespace(pixel_formats=[], sizes=lambda _p: [])
                )
            )
        )

    def create_preview_configuration(
        self,
        main: Optional[Dict[str, Any]] = None,
        controls: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        """Build an opaque config recording the ``main`` size and ScalerCrop."""
        size = tuple(main["size"]) if main and "size" in main else _INPUT_SIZE
        scaler = controls.get("ScalerCrop") if controls else None
        return {"main_size": (int(size[0]), int(size[1])), "scaler": scaler}

    def _adopt(self, config: Optional[Dict[str, Any]]) -> None:
        """Adopt the ``main`` size + ScalerCrop from a config dict, if present."""
        if not isinstance(config, dict):
            return
        if config.get("main_size"):
            self.main_size = config["main_size"]
        if config.get("scaler"):
            self._scaler = tuple(config["scaler"])  # type: ignore[assignment]

    def start(
        self, config: Optional[Dict[str, Any]] = None, show_preview: bool = False
    ) -> None:
        """Start the camera, adopting the given config's geometry."""
        del show_preview
        self._adopt(config)
        self.started = True

    def stop(self) -> None:
        """Stop the camera."""
        self.started = False

    def configure(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Reconfigure the camera geometry from a config dict."""
        self._adopt(config)

    def set_controls(self, controls: Dict[str, Any]) -> None:
        """Apply controls; a new ScalerCrop is recorded for later requests."""
        if "ScalerCrop" in controls:
            self._scaler = tuple(controls["ScalerCrop"])  # type: ignore[assignment]

    def camera_configuration(self) -> Dict[str, Any]:
        """Return a minimal configuration (raw format for the full-FOV path)."""
        return {"raw": {"format": "SBGGR10"}}

    def _next_main_frame(self) -> np.ndarray:
        """Pull the next source frame and resize it to the ``main`` stream size.

        Raises:
            KeyboardInterrupt: When the frame source is exhausted, so the real
                capture loop unwinds through ``main``'s existing shutdown path.
        """
        raw = self._state.frame_source.next_frame()
        if raw is None:
            raise KeyboardInterrupt("Emulated frame source exhausted")
        width, height = self.main_size
        return cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)

    def capture_metadata(self) -> Dict[str, Any]:
        """Advance one frame: detect, fire ``pre_callback``, return metadata.

        Raises:
            KeyboardInterrupt: When the configured ``max_frames`` is reached or
                the frame source is exhausted (both unwind the real loop via
                ``main``'s ``except KeyboardInterrupt`` shutdown).
        """
        frame = self._next_main_frame()
        self._last_frame = frame
        self._state.imx500.current_detections = self._state.detector.detect(frame)

        request = FakeRequest(frame, self._scaler)
        if self.pre_callback is not None:
            self.pre_callback(request)

        self._frame_count += 1
        if (
            self._state.max_frames is not None
            and self._frame_count >= self._state.max_frames
        ):
            raise KeyboardInterrupt("Emulated frame budget reached")
        return {"FrameId": self._frame_count}

    def capture_request(self) -> FakeRequest:
        """Return a request for the most recent (or a freshly pulled) frame."""
        frame = (
            self._last_frame
            if self._last_frame is not None
            else self._next_main_frame()
        )
        return FakeRequest(frame, self._scaler)

    def capture_array(self, stream: str = "main") -> np.ndarray:
        """Return the most recent ``main`` frame (or a fresh one)."""
        del stream
        if self._last_frame is not None:
            return self._last_frame
        return self._next_main_frame()


class FakeTransform:
    """Stand-in for ``libcamera.Transform`` (records the flip flags)."""

    def __init__(self, vflip: bool = False, hflip: bool = False) -> None:
        """Record the requested vertical/horizontal flips."""
        self.vflip = vflip
        self.hflip = hflip


class FakeSensorFormat:
    """Stand-in for ``picamera2.sensor_format.SensorFormat`` (full-FOV path only)."""

    def __init__(self, name: str) -> None:
        """Record the format name."""
        self.name = name

    @property
    def unpacked(self) -> str:
        """Return the (unchanged) format name as its 'unpacked' variant."""
        return self.name


@dataclass
class _EmulationState:
    """The live emulation wiring shared by the fake camera objects.

    Attributes:
        frame_source: Where the emulated camera pulls frames from.
        detector: The off-Pi object detector run per frame.
        imx500: The single shared fake IMX500 the fakes read/write.
        max_frames: Stop after this many frames (``None`` runs until the source
            is exhausted); used to bound tests.
    """

    frame_source: FrameSource
    detector: Detector
    imx500: FakeIMX500 = field(default_factory=FakeIMX500)
    max_frames: Optional[int] = None


_STATE: Optional[_EmulationState] = None


def set_emulation_state(state: Optional[_EmulationState]) -> None:
    """Install (or clear) the active emulation state the fakes read from."""
    global _STATE
    _STATE = state


def _require_state() -> _EmulationState:
    """Return the active emulation state or raise if none is installed."""
    if _STATE is None:
        raise RuntimeError(
            "No emulation state installed; call install_fake_camera_modules first"
        )
    return _STATE


def build_emulation_state(
    frame_source: FrameSource,
    detector: Detector,
    *,
    max_frames: Optional[int] = None,
) -> _EmulationState:
    """Build an :class:`_EmulationState` and install it as the active state."""
    state = _EmulationState(
        frame_source=frame_source, detector=detector, max_frames=max_frames
    )
    set_emulation_state(state)
    return state


def active_imx500() -> FakeIMX500:
    """Return the shared fake IMX500 from the active emulation state."""
    return _require_state().imx500
