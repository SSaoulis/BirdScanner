"""Off-Pi dev suite for the stable-box tracking + classification method.

Takes a recorded ``.mp4`` in and produces a composite "dashboard" ``.mp4`` out:
each output frame is the annotated video frame on top (tracked boxes, ``#id`` +
stability count, gold once stable, species once classified) with two time-series
plots stacked below it — bird-class confidence over time (``0`` on frames with no
detection) and the tracked ``n_stable`` frame count over time — each carrying a
live cursor at the current frame. The classifier's best frame per track is shown
as a panel appended at the end.

The pipeline stands in for the Pi's IMX500 with the off-Pi ``OnnxYoloDetector``
(a stock YOLO11n ONNX, the same detector the camera emulator uses); everything
downstream is the *real* code from :mod:`birdscanner.ml` — the ``StableDetectionTracker``,
``BestFrameSelector``, ``preprocess_roi`` and the ConvNeXt ``Classifier`` — so
whatever tracking/classification method you prototype here moves straight into
the live pipeline.

The reusable building blocks (:func:`run_review`, :func:`render_dashboard_video`,
:func:`annotate_frame`, :class:`ReviewConfig`) are shared with
``notebooks/tracking_playground.ipynb`` and exposed as a CLI::

    python -m dev.tracking_review INPUT.mp4 OUTPUT.mp4

Requires the out-of-band ``assets/models/yolo11n.onnx`` (export once with
``yolo export model=yolo11n.pt format=onnx``). Species classification is optional:
when the classifier ONNX is absent the suite runs detection + tracking only and
labels boxes ``bird`` with the detection confidence.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from birdscanner.ml.best_frame import BestCandidate, BestFrameSelector
from birdscanner.ml.classification import Classifier
from birdscanner.ml.classification_pipeline import setup_classifier
from birdscanner.ml.detection_utils import preprocess_roi
from birdscanner.ml.tracking import StableDetectionTracker, StableTrack
from dev.emulation.yolo import OnnxYoloDetector

logger = logging.getLogger(__name__)

# Field-journal palette, matching the notebook's existing colours (RGB tuples —
# frames stay RGB end-to-end and are only converted to BGR at the VideoWriter).
_STABLE_RGB = (200, 138, 26)  # gold: a stable track
_TRACKING_RGB = (59, 122, 87)  # green: a track not yet stable
_CURSOR_RGB = "#8a8a8a"

# Repo root is the parent of the ``dev/`` package this module lives in.
REPO_ROOT = Path(__file__).resolve().parents[1]

TrackerFactory = Callable[
    [Optional[Callable[[StableTrack], None]], Optional[Callable[[StableTrack], None]]],
    StableDetectionTracker,
]


@dataclass
class Det:
    """A bird detection in the shape the tracker + classifier consume.

    ``box`` is pixel ``(x, y, w, h)``; ``category``/``label`` mirror the fields the
    real ``Detection`` carries so a ``Det`` can be fed straight to ``update_frame``.

    Attributes:
        box: Bounding box ``(x, y, w, h)`` in frame pixel coordinates.
        conf: Object-detection (YOLO) confidence in ``[0, 1]``.
        category: Object-detection class index (unused downstream; kept for parity).
        label: COCO class name, e.g. ``"bird"``.
    """

    box: tuple
    conf: float
    category: int
    label: str


@dataclass
class ReviewConfig:
    """Tuning + model resolution for a review run.

    Bundled into one value object so the pipeline/render functions keep a short
    parameter list. Model paths are resolved lazily by :func:`resolve_yolo_model`
    and :func:`build_classifier` so an absent classifier degrades gracefully.

    Attributes:
        detector_conf: Object-detection confidence floor.
        iou_threshold: IoU to match a detection to an existing track.
        min_stable_frames: Frames a track must persist before it is "stable".
        max_missing_frames: Frames a track survives unmatched (0 = drop at once).
        target_label: The only COCO class kept for tracking.
        classify: Whether to run the ConvNeXt classifier on each best frame.
        yolo_model_path: Explicit YOLO ONNX path; ``None`` triggers auto-resolution.
        classifier_model_path: ConvNeXt ONNX path; absent → detection-only.
        class_to_idx_path: Classifier class-to-index JSON.
        best_frame_tail_seconds: Seconds to hold the best-frame panel at the end.
    """

    detector_conf: float = 0.25
    iou_threshold: float = 0.3
    min_stable_frames: int = 3
    max_missing_frames: int = 30
    target_label: str = "bird"
    classify: bool = True
    yolo_model_path: Optional[str] = None
    classifier_model_path: str = str(
        REPO_ROOT / "assets" / "models" / "convnext_v2_tiny_int8.onnx"
    )
    class_to_idx_path: str = str(
        REPO_ROOT / "assets" / "models" / "convnext_v2_tiny.onnx_class_to_idx.json"
    )
    best_frame_tail_seconds: float = 2.5


@dataclass
class TrackView:
    """A single track's render state for one frame (no pixel data retained).

    Attributes:
        box: Detection box ``(x, y, w, h)`` in frame pixels.
        track_id: Stable track identifier.
        stable_frames: How many consecutive frames the track has held.
        classified: Whether the track has been classified yet.
        conf: The detection confidence for this frame.
        species: Classified species once known, else ``None``.
    """

    box: tuple
    track_id: int
    stable_frames: int
    classified: bool
    conf: float
    species: Optional[str] = None


@dataclass
class FrameStats:
    """Per-frame aggregates for the dashboard plots.

    Attributes:
        top_conf: Highest bird detection confidence this frame (``0.0`` if none).
        n_stable: Highest ``stable_frames`` across active tracks (``0`` if none).
        tracks: The per-track render records for this frame.
    """

    top_conf: float
    n_stable: int
    tracks: list[TrackView]


@dataclass
class BestFrameView:
    """The best frame a track fed to the classifier, plus its verdict.

    Attributes:
        candidate: The retained highest-confidence frame + box for the track.
        species: Classified species, or ``None`` when classification was skipped.
        confidence: Classifier confidence, or the detection confidence when skipped.
    """

    candidate: BestCandidate
    species: Optional[str]
    confidence: float


@dataclass
class ReviewResult:
    """Everything a review run produced, consumed by the renderer + notebook.

    Attributes:
        fps: Source video frame rate (used for the output clip + tail length).
        frame_count: Number of frames processed.
        min_stable_frames: The tracker's stability threshold (for colouring).
        per_frame: One :class:`FrameStats` per processed frame, in order.
        best_frames: Best classified frame per track id.
        events: ``{"stable": [...], "deleted": [...]}`` lifecycle events, each
            entry ``(frame_index, track_id, stable_frames)``.
        frames: The RGB frames, only when ``run_review(..., keep_frames=True)``.
    """

    fps: float
    frame_count: int
    min_stable_frames: int
    per_frame: list[FrameStats] = field(default_factory=list)
    best_frames: dict[int, BestFrameView] = field(default_factory=dict)
    events: dict[str, list] = field(
        default_factory=lambda: {"stable": [], "deleted": []}
    )
    frames: Optional[list[np.ndarray]] = None


def norm_xyxy_to_pixel_xywh(box: tuple, width: int, height: int) -> tuple:
    """Convert a normalized ``(x0, y0, x1, y1)`` box to pixel ``(x, y, w, h)``.

    Args:
        box: Normalized ``(x0, y0, x1, y1)`` fractions of the frame.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        The box as pixel ``(x, y, w, h)``.
    """
    x0, y0, x1, y1 = box
    x, y = x0 * width, y0 * height
    return (x, y, (x1 - x0) * width, (y1 - y0) * height)


def resolve_yolo_model(config: ReviewConfig) -> Path:
    """Resolve the YOLO ONNX model path from config, env, then known locations.

    Checks, first-existing-wins: ``config.yolo_model_path``, the ``YOLO_ONNX_PATH``
    env var, ``assets/models/yolo11n.onnx``, then a project-root ``yolo11n.onnx``.

    Args:
        config: The review configuration.

    Returns:
        The path to an existing YOLO ONNX model.

    Raises:
        FileNotFoundError: If no candidate path exists.
    """
    candidates = [
        config.yolo_model_path,
        os.environ.get("YOLO_ONNX_PATH"),
        str(REPO_ROOT / "assets" / "models" / "yolo11n.onnx"),
        str(REPO_ROOT / "yolo11n.onnx"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "No YOLO ONNX model found. Set --yolo-model / YOLO_ONNX_PATH, or place "
        "yolo11n.onnx in assets/models/ (export it once with "
        "`yolo export model=yolo11n.pt format=onnx`)."
    )


def build_classifier(config: ReviewConfig) -> Optional[Classifier]:
    """Build the ConvNeXt classifier, or ``None`` when it should be skipped.

    Returns ``None`` (and logs why) when classification is disabled or the model /
    class-map files are absent, so the suite degrades to detection + tracking only.

    Args:
        config: The review configuration.

    Returns:
        A ready :class:`Classifier`, or ``None``.
    """
    if not config.classify:
        return None
    model = Path(config.classifier_model_path)
    class_map = Path(config.class_to_idx_path)
    if not model.is_file() or not class_map.is_file():
        logger.warning(
            "Classifier model absent (%s); running detection + tracking only.", model
        )
        return None
    return setup_classifier(str(model), str(class_map))


def detect_birds(
    detector: OnnxYoloDetector, frame: np.ndarray, target_label: str
) -> list[Det]:
    """Detect the target class in one RGB frame as pixel-space ``Det`` objects.

    Args:
        detector: The off-Pi YOLO detector (IMX500 stand-in).
        frame: RGB ``(H, W, 3)`` uint8 frame.
        target_label: COCO class name to keep, e.g. ``"bird"``.

    Returns:
        The kept detections as ``Det`` objects in pixel ``(x, y, w, h)``.
    """
    height, width = frame.shape[:2]
    return [
        Det(norm_xyxy_to_pixel_xywh(d.box, width, height), d.score, 0, d.label)
        for d in detector.detect(frame)
        if d.label == target_label
    ]


def default_tracker_factory(config: ReviewConfig) -> TrackerFactory:
    """Return a factory building the current pipeline tracker from ``config``.

    Args:
        config: The review configuration supplying the tracker parameters.

    Returns:
        A ``(on_stable, on_deleted) -> StableDetectionTracker`` factory.
    """

    def factory(
        on_stable: Optional[Callable[[StableTrack], None]],
        on_deleted: Optional[Callable[[StableTrack], None]],
    ) -> StableDetectionTracker:
        """Build the tracker wired with the given lifecycle callbacks."""
        return StableDetectionTracker(
            iou_threshold=config.iou_threshold,
            min_stable_frames=config.min_stable_frames,
            max_missing_frames=config.max_missing_frames,
            on_track_became_stable=on_stable,
            on_track_deleted=on_deleted,
        )

    return factory


@dataclass
class _RunState:
    """Mutable per-run state, keeping :func:`run_review`'s local count low."""

    config: ReviewConfig
    detector: OnnxYoloDetector
    tracker: StableDetectionTracker
    selector: BestFrameSelector
    classifier: Optional[Classifier]
    min_stable_frames: int
    result: ReviewResult
    current_index: int = 0


def _classify_best_frame(state: _RunState, track: StableTrack) -> None:
    """Classify a newly stable track's best frame and record the verdict.

    Mirrors the live pipeline: take the track's best (highest-confidence) frame,
    crop the padded-square classifier ROI, and — when a classifier is wired and the
    ROI is non-empty — predict the species. A zero-area ROI is re-offered so the
    track can classify on a later frame instead of being lost.

    Args:
        state: The active run state.
        track: The track that just crossed the stability threshold.
    """
    best = state.selector.take(track.track_id)
    if best is None:
        return
    roi, _ = preprocess_roi(best.frame, best.box)
    if roi.size == 0:
        state.selector.observe(track.track_id, best.frame, best.box, best.score)
        return
    species: Optional[str] = None
    confidence = best.score
    if state.classifier is not None:
        species, confidence = state.classifier.classify(roi)
    state.tracker.mark_classified(track.track_id, species=species)
    state.result.best_frames[track.track_id] = BestFrameView(
        candidate=best, species=species, confidence=confidence
    )


def _process_frame(state: _RunState, frame: np.ndarray) -> None:
    """Detect, track, gate classification and record stats for one frame.

    Args:
        state: The active run state.
        frame: RGB ``(H, W, 3)`` uint8 frame.
    """
    detections = detect_birds(state.detector, frame, state.config.target_label)
    state.tracker.update_frame(detections)

    views: list[TrackView] = []
    top_conf = 0.0
    n_stable = 0
    for det_id, det in enumerate(detections):
        track = state.tracker.track_for_detection_id(det_id)
        if track is None:
            continue
        state.selector.observe(track.track_id, frame, det.box, det.conf)
        top_conf = max(top_conf, float(det.conf))
        n_stable = max(n_stable, track.stable_frames)
        if not track.classified and track.stable_frames >= state.min_stable_frames:
            _classify_best_frame(state, track)
        views.append(
            TrackView(
                box=det.box,
                track_id=track.track_id,
                stable_frames=track.stable_frames,
                classified=track.classified,
                conf=float(det.conf),
                species=track.species,
            )
        )
    state.result.per_frame.append(
        FrameStats(top_conf=top_conf, n_stable=n_stable, tracks=views)
    )


def run_review(
    video_path: str,
    config: ReviewConfig,
    *,
    make_tracker: Optional[TrackerFactory] = None,
    keep_frames: bool = False,
) -> ReviewResult:
    """Run detect -> track -> best-frame -> classify over every frame of a clip.

    The detector stands in for the IMX500; the tracker, best-frame selector and
    classifier are the real pipeline code. Pass ``make_tracker`` to prototype a new
    tracking method (any object exposing ``update_frame`` +
    ``track_for_detection_id`` + ``mark_classified``). ``keep_frames`` retains the
    RGB frames on the result for fast in-notebook rendering; leave it ``False`` for
    the CLI so RAM stays flat (the renderer re-reads frames from disk).

    Args:
        video_path: Path to the input ``.mp4``.
        config: Tuning + model resolution.
        make_tracker: Optional tracker factory; defaults to the pipeline tracker.
        keep_frames: Retain RGB frames on the result when ``True``.

    Returns:
        A populated :class:`ReviewResult`.
    """
    detector = OnnxYoloDetector(
        str(resolve_yolo_model(config)), conf_threshold=config.detector_conf
    )
    cap = cv2.VideoCapture(video_path)
    result = ReviewResult(
        fps=cap.get(cv2.CAP_PROP_FPS) or 15.0,
        frame_count=0,
        min_stable_frames=config.min_stable_frames,
        frames=[] if keep_frames else None,
    )
    state = _build_run_state(config, detector, result, make_tracker)
    _run_capture_loop(cap, state)
    return result


def _build_run_state(
    config: ReviewConfig,
    detector: OnnxYoloDetector,
    result: ReviewResult,
    make_tracker: Optional[TrackerFactory],
) -> _RunState:
    """Assemble the tracker (wired to lifecycle callbacks) + per-run state.

    Args:
        config: The review configuration.
        detector: The off-Pi YOLO detector.
        result: The result object the callbacks append lifecycle events to.
        make_tracker: Optional tracker factory; defaults to the pipeline tracker.

    Returns:
        The initialised :class:`_RunState`.
    """
    selector = BestFrameSelector()

    def on_stable(track: StableTrack) -> None:
        result.events["stable"].append(
            (result.frame_count, track.track_id, track.stable_frames)
        )

    def on_deleted(track: StableTrack) -> None:
        result.events["deleted"].append(
            (result.frame_count, track.track_id, track.stable_frames)
        )
        selector.discard(track.track_id)

    tracker = (make_tracker or default_tracker_factory(config))(on_stable, on_deleted)
    result.min_stable_frames = getattr(
        tracker, "min_stable_frames", config.min_stable_frames
    )
    return _RunState(
        config=config,
        detector=detector,
        tracker=tracker,
        selector=selector,
        classifier=build_classifier(config),
        min_stable_frames=result.min_stable_frames,
        result=result,
    )


def _run_capture_loop(cap: cv2.VideoCapture, state: _RunState) -> None:
    """Drive every source frame through the detect/track/classify pipeline.

    Args:
        cap: The opened video capture (released here).
        state: The active run state (its ``result`` is populated in place).
    """
    result = state.result
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            state.current_index = result.frame_count
            _process_frame(state, frame)
            if result.frames is not None:
                result.frames.append(frame)
            result.frame_count += 1
    finally:
        cap.release()


def annotate_frame(
    frame: np.ndarray, tracks: list[TrackView], min_stable_frames: int
) -> np.ndarray:
    """Draw each track's box + ``#id`` / stability / species onto a frame copy.

    Boxes are gold once the track is stable, green before; the species is appended
    to the caption once the track has been classified.

    Args:
        frame: RGB frame to annotate (not mutated; a copy is returned).
        tracks: The per-track render records for this frame.
        min_stable_frames: Stability threshold used to pick the box colour.

    Returns:
        A new annotated RGB frame.
    """
    out = frame.copy()
    for track in tracks:
        x, y, w, h = (int(v) for v in track.box)
        color = (
            _STABLE_RGB if track.stable_frames >= min_stable_frames else _TRACKING_RGB
        )
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        caption = f"#{track.track_id} s{track.stable_frames}"
        if track.species:
            caption += f" {track.species}"
        cv2.putText(
            out,
            caption,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return out


@dataclass
class _SeriesStyle:
    """Styling for one dashboard plot, keeping :func:`_plot_series` args short."""

    color: str
    ylabel: str
    ylim: Optional[tuple]


@dataclass
class _DashboardLayout:
    """Fixed figure geometry so every composite frame is the same pixel size.

    Attributes:
        fig_w_in: Figure width in inches.
        fig_h_in: Figure height in inches.
        dpi: Dots per inch.
        height_ratios: Row height ratios ``(video, conf_plot, stable_plot)``.
        n_frames: Total frames, for the plots' x-axis extent.
    """

    fig_w_in: float
    fig_h_in: float
    dpi: int
    height_ratios: tuple
    n_frames: int


def _build_layout(frame_shape: tuple, n_frames: int) -> _DashboardLayout:
    """Derive a constant dashboard layout from the frame aspect ratio.

    Args:
        frame_shape: ``(H, W, ...)`` shape of a source frame.
        n_frames: Total number of frames, for the plot x-axis.

    Returns:
        A :class:`_DashboardLayout` yielding an even-sized RGBA canvas.
    """
    dpi = 100
    width_px = 1000
    height, width = frame_shape[:2]
    top_px = int(round(width_px * height / width))
    plots_px = 320
    total_px = top_px + plots_px
    if total_px % 2:  # keep the encoder happy with even dimensions
        top_px += 1
        total_px += 1
    return _DashboardLayout(
        fig_w_in=width_px / dpi,
        fig_h_in=total_px / dpi,
        dpi=dpi,
        height_ratios=(top_px, plots_px / 2, plots_px / 2),
        n_frames=max(1, n_frames),
    )


def _plot_series(
    ax: Axes, series: list, cursor: int, style: _SeriesStyle, n_frames: int
) -> None:
    """Draw one time-series with a vertical cursor at the current frame.

    Args:
        ax: The axis to draw on.
        series: The per-frame values.
        cursor: The current frame index (cursor position).
        style: Colour / label / y-limit styling.
        n_frames: Total frame count for the x-axis extent.
    """
    xs = range(len(series))
    ax.plot(xs, series, color=style.color, linewidth=1.3)
    ax.axvline(cursor, color=_CURSOR_RGB, linewidth=1.0)
    if 0 <= cursor < len(series):
        ax.plot(cursor, series[cursor], "o", color=style.color, markersize=4)
    ax.set_xlim(0, n_frames - 1)
    if style.ylim is not None:
        ax.set_ylim(*style.ylim)
    ax.set_ylabel(style.ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def _composite_frame(
    top_rgb: np.ndarray,
    series: tuple,
    cursor: int,
    layout: _DashboardLayout,
) -> np.ndarray:
    """Render one dashboard frame (video pane + two plots) as a BGR array.

    Args:
        top_rgb: The already-annotated RGB frame for the top pane.
        series: ``(conf_series, stable_series)`` for the two plots.
        cursor: The current frame index.
        layout: The fixed figure geometry.

    Returns:
        A BGR ``(H, W, 3)`` uint8 frame ready for ``VideoWriter.write``.
    """
    conf_series, stable_series = series
    fig = Figure(figsize=(layout.fig_w_in, layout.fig_h_in), dpi=layout.dpi)
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 1, height_ratios=layout.height_ratios, hspace=0.55)

    ax_img = fig.add_subplot(grid[0])
    ax_img.imshow(top_rgb)
    ax_img.axis("off")

    ax_conf = fig.add_subplot(grid[1])
    _plot_series(
        ax_conf,
        conf_series,
        cursor,
        _SeriesStyle("#c58a1a", "bird conf", (0.0, 1.05)),
        layout.n_frames,
    )
    ax_stable = fig.add_subplot(grid[2])
    _plot_series(
        ax_stable,
        stable_series,
        cursor,
        _SeriesStyle("#3b7a57", "n_stable", None),
        layout.n_frames,
    )
    ax_stable.set_xlabel("frame", fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.99, bottom=0.09)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return cv2.cvtColor(np.ascontiguousarray(rgba[:, :, :3]), cv2.COLOR_RGB2BGR)


def _open_writer(out_path: str, size: tuple, fps: float) -> cv2.VideoWriter:
    """Open an mp4 writer, preferring browser-playable H.264, else ``mp4v``.

    Args:
        out_path: Destination ``.mp4`` path.
        size: ``(width, height)`` in pixels.
        fps: Frames per second.

    Returns:
        An opened :class:`cv2.VideoWriter`.
    """
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not writer.isOpened():
        logger.warning(
            "H.264 (avc1) unavailable; falling back to mp4v (may not play in-browser)."
        )
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return writer


def _frame_supply(video_path: str, result: ReviewResult) -> Iterator[np.ndarray]:
    """Yield RGB frames from the result cache, else by re-reading the source.

    Args:
        video_path: The input clip (re-read when frames were not cached).
        result: The review result (may carry cached frames).

    Yields:
        RGB ``(H, W, 3)`` frames in order.
    """
    if result.frames is not None:
        yield from result.frames
        return
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _best_frame_panel(
    view: BestFrameView, track_id: int, min_stable_frames: int
) -> np.ndarray:
    """Build the top-pane image for a track's best frame + classifier ROI inset.

    Args:
        view: The best frame + verdict for the track.
        track_id: The track's identifier.
        min_stable_frames: Stability threshold (best frame is always drawn gold).

    Returns:
        An annotated RGB frame with the ROI inset top-left.
    """
    label = view.species or "bird"
    caption = TrackView(
        box=view.candidate.box,
        track_id=track_id,
        stable_frames=min_stable_frames,
        classified=True,
        conf=view.candidate.score,
        species=f"{label} ({view.confidence:.2f})",
    )
    panel = annotate_frame(view.candidate.frame, [caption], min_stable_frames)
    roi, _ = preprocess_roi(view.candidate.frame, view.candidate.box)
    height, width = panel.shape[:2]
    # Inset the classifier ROI top-left, sized to fit even small frames.
    size = min(160, width // 3, height // 3)
    if roi.size and size >= 8:
        inset = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
        panel[10 : 10 + size, 10 : 10 + size] = inset
        cv2.rectangle(panel, (10, 10), (10 + size, 10 + size), _STABLE_RGB, 2)
    return panel


def _append_best_frame_tail(
    writer: cv2.VideoWriter,
    result: ReviewResult,
    config: ReviewConfig,
    layout: _DashboardLayout,
) -> None:
    """Append the best-frame panels (one slice of the tail per track) to the clip.

    Args:
        writer: The open video writer.
        result: The review result carrying the best frames.
        config: Supplies the tail duration.
        layout: The fixed figure geometry.
    """
    if not result.best_frames:
        return
    series = (
        [fs.top_conf for fs in result.per_frame],
        [fs.n_stable for fs in result.per_frame],
    )
    cursor = len(result.per_frame) - 1
    tail_total = int(round(config.best_frame_tail_seconds * result.fps))
    per_track = max(1, tail_total // len(result.best_frames))
    for track_id, view in result.best_frames.items():
        panel = _best_frame_panel(view, track_id, result.min_stable_frames)
        composite = _composite_frame(panel, series, cursor, layout)
        for _ in range(per_track):
            writer.write(composite)


def render_dashboard_video(
    video_path: str, result: ReviewResult, out_path: str, config: ReviewConfig
) -> None:
    """Render the composite dashboard ``.mp4`` from a completed review run.

    Each output frame stacks the annotated video frame over the confidence and
    ``n_stable`` time-series (with a live cursor); the best classified frame per
    track is appended as a panel at the end.

    Args:
        video_path: The input clip (re-read when the result has no cached frames).
        result: The completed :class:`ReviewResult`.
        out_path: Destination ``.mp4`` path.
        config: Supplies the stability threshold + tail duration.
    """
    if not result.per_frame:
        raise ValueError("ReviewResult has no frames to render.")
    series = (
        [fs.top_conf for fs in result.per_frame],
        [fs.n_stable for fs in result.per_frame],
    )
    layout: Optional[_DashboardLayout] = None
    writer: Optional[cv2.VideoWriter] = None
    for idx, frame in enumerate(_frame_supply(video_path, result)):
        if idx >= len(result.per_frame):
            break
        annotated = annotate_frame(
            frame, result.per_frame[idx].tracks, result.min_stable_frames
        )
        if layout is None:
            layout = _build_layout(frame.shape, len(result.per_frame))
        composite = _composite_frame(annotated, series, idx, layout)
        if writer is None:
            writer = _open_writer(
                out_path, (composite.shape[1], composite.shape[0]), result.fps
            )
        writer.write(composite)

    if writer is not None and layout is not None:
        _append_best_frame_tail(writer, result, config, layout)
        writer.release()


def _build_config_from_args(args: argparse.Namespace) -> ReviewConfig:
    """Translate parsed CLI arguments into a :class:`ReviewConfig`.

    Args:
        args: The parsed argparse namespace.

    Returns:
        The configured :class:`ReviewConfig`.
    """
    config = ReviewConfig(
        detector_conf=args.conf,
        iou_threshold=args.iou,
        min_stable_frames=args.min_stable,
        max_missing_frames=args.max_missing,
        classify=not args.no_classify,
        yolo_model_path=args.yolo_model,
        best_frame_tail_seconds=args.best_frame_seconds,
    )
    if args.classifier_model:
        config.classifier_model_path = args.classifier_model
    return config


def _parse_args(argv: Optional[list]) -> argparse.Namespace:
    """Parse the command-line arguments for the dashboard CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m dev.tracking_review",
        description="Render a tracking + classification dashboard mp4 from a clip.",
    )
    parser.add_argument("input", help="Input .mp4 clip")
    parser.add_argument("output", help="Output dashboard .mp4 path")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Detection confidence floor"
    )
    parser.add_argument(
        "--iou", type=float, default=0.3, help="Tracker IoU match threshold"
    )
    parser.add_argument(
        "--min-stable", type=int, default=3, help="Frames to become stable"
    )
    parser.add_argument(
        "--max-missing", type=int, default=30, help="Frames a track survives unmatched"
    )
    parser.add_argument(
        "--no-classify", action="store_true", help="Skip species classification"
    )
    parser.add_argument(
        "--best-frame-seconds",
        type=float,
        default=2.5,
        help="Best-frame panel duration",
    )
    parser.add_argument("--yolo-model", default=None, help="Override YOLO ONNX path")
    parser.add_argument(
        "--classifier-model", default=None, help="Override classifier ONNX path"
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """CLI entry point: run a review and render the dashboard clip.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    config = _build_config_from_args(args)

    logger.info("Running detection + tracking over %s ...", args.input)
    result = run_review(args.input, config)
    logger.info(
        "Processed %d frames (%.1f fps); %d track(s) classified.",
        result.frame_count,
        result.fps,
        len(result.best_frames),
    )
    for track_id, view in result.best_frames.items():
        logger.info(
            "  track #%d -> %s (%.2f)",
            track_id,
            view.species or "bird",
            view.confidence,
        )

    logger.info("Rendering dashboard to %s ...", args.output)
    render_dashboard_video(args.input, result, args.output, config)
    logger.info("Done: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
