"""Tests for the off-Pi tracking + classification dev suite (``dev/tracking_review``).

The pure helpers (conversion, annotation, model resolution, graceful classifier
fallback) and the whole render path run with no model files. The full
``run_review`` integration test is skipped when ``yolo11n.onnx`` (out-of-band) or
the test clip is absent, matching the model-absent skip pattern used elsewhere.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from birdscanner.ml.best_frame import BestCandidate
from dev.tracking_review import (
    BestFrameView,
    FrameStats,
    ReviewConfig,
    ReviewResult,
    TrackView,
    annotate_frame,
    build_classifier,
    norm_xyxy_to_pixel_xywh,
    render_dashboard_video,
    resolve_yolo_model,
    run_review,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_YOLO_MODEL = _REPO_ROOT / "assets" / "models" / "yolo11n.onnx"
_TEST_VIDEO = _REPO_ROOT / "tests" / "_test_videos" / "great_tit_2.mp4"

_STABLE_RGB = (200, 138, 26)
_TRACKING_RGB = (59, 122, 87)


def test_norm_xyxy_to_pixel_xywh_converts():
    """A normalized xyxy box maps to the expected pixel (x, y, w, h)."""
    assert norm_xyxy_to_pixel_xywh((0.1, 0.2, 0.5, 0.7), 100, 200) == pytest.approx(
        (10.0, 40.0, 40.0, 100.0)
    )


def test_annotate_frame_colours_by_stability():
    """Stable tracks draw gold, not-yet-stable tracks green; input is untouched."""
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    tracks = [
        TrackView(
            box=(5, 5, 20, 20), track_id=1, stable_frames=1, classified=False, conf=0.5
        ),
        TrackView(
            box=(40, 40, 20, 20),
            track_id=2,
            stable_frames=5,
            classified=True,
            conf=0.9,
            species="Robin",
        ),
    ]
    out = annotate_frame(frame, tracks, min_stable_frames=3)

    assert out is not frame
    assert int(frame.max()) == 0  # the original frame is not mutated
    colours = {tuple(int(c) for c in px) for row in out for px in row}
    assert _STABLE_RGB in colours
    assert _TRACKING_RGB in colours


def test_resolve_yolo_model_prefers_explicit_path(tmp_path):
    """An explicit, existing model path wins over the fallback locations."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"stub")
    assert resolve_yolo_model(ReviewConfig(yolo_model_path=str(model))) == model


def test_build_classifier_absent_returns_none(tmp_path):
    """A missing classifier model degrades to None (detection + tracking only)."""
    config = ReviewConfig(classifier_model_path=str(tmp_path / "nope.onnx"))
    assert build_classifier(config) is None


def test_build_classifier_disabled_returns_none():
    """classify=False never loads a classifier."""
    assert build_classifier(ReviewConfig(classify=False)) is None


def _encoder_available(path: Path) -> bool:
    """Return True if this OpenCV build can open an mp4 writer."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 12))
    opened = writer.isOpened()
    writer.release()
    return opened


def _synthetic_result(n_frames: int = 5) -> ReviewResult:
    """Build a small in-memory ReviewResult (with cached frames) for render tests."""
    frames = [np.full((60, 80, 3), 30, dtype=np.uint8) for _ in range(n_frames)]
    per_frame = [
        FrameStats(
            top_conf=0.4 + 0.1 * i,
            n_stable=i + 1,
            tracks=[
                TrackView(
                    box=(10, 10, 20, 20),
                    track_id=0,
                    stable_frames=i + 1,
                    classified=i >= 3,
                    conf=0.4 + 0.1 * i,
                    species="Robin" if i >= 3 else None,
                )
            ],
        )
        for i in range(n_frames)
    ]
    best = {
        0: BestFrameView(
            candidate=BestCandidate(frame=frames[-1], box=(10, 10, 20, 20), score=0.8),
            species="Robin",
            confidence=0.77,
        )
    }
    return ReviewResult(
        fps=5.0,
        frame_count=n_frames,
        min_stable_frames=3,
        per_frame=per_frame,
        best_frames=best,
        events={"stable": [(3, 0, 3)], "deleted": []},
        frames=frames,
    )


def test_render_dashboard_writes_video(tmp_path):
    """The composite dashboard renders to a non-empty mp4 from cached frames."""
    if not _encoder_available(tmp_path / "probe.mp4"):
        pytest.skip("no mp4 encoder available in this OpenCV build")
    out = tmp_path / "dash.mp4"
    render_dashboard_video("unused.mp4", _synthetic_result(), str(out), ReviewConfig())
    assert out.exists() and out.stat().st_size > 0


def test_render_dashboard_rejects_empty_result(tmp_path):
    """Rendering a result with no frames is a programming error."""
    empty = ReviewResult(fps=5.0, frame_count=0, min_stable_frames=3)
    with pytest.raises(ValueError):
        render_dashboard_video("x.mp4", empty, str(tmp_path / "y.mp4"), ReviewConfig())


@pytest.mark.skipif(
    not (_YOLO_MODEL.exists() and _TEST_VIDEO.exists()),
    reason="yolo11n.onnx or test video absent",
)
def test_run_review_end_to_end():
    """run_review produces one FrameStats per frame with valid, bounded stats."""
    result = run_review(str(_TEST_VIDEO), ReviewConfig(classify=False))
    assert result.frame_count > 0
    assert len(result.per_frame) == result.frame_count
    assert any(fs.tracks for fs in result.per_frame)
    assert all(0.0 <= fs.top_conf <= 1.0 for fs in result.per_frame)
    assert all(fs.n_stable >= 0 for fs in result.per_frame)
