"""Unit tests for VideoRecorder buffer + trigger logic.

Frames come from the shared ``frame_factory`` fixture. The actual mp4 encode
depends on the OpenCV FFmpeg backend, which is not guaranteed in every CI
environment, so the handoff-to-encode is exercised by monkeypatching ``_encode``.
One best-effort test drives the real encoder but only asserts on the file when the
backend produced one.
"""

import threading

from birdscanner.detector.video_recorder import VideoRecorder


def test_pre_roll_buffer_is_bounded(frame_factory):
    """The ring buffer keeps at most fps * pre_roll_seconds frames."""
    rec = VideoRecorder(fps=10, pre_roll_seconds=1.0, post_roll_seconds=1.0)
    for i in range(25):
        rec.add_frame(frame_factory(i, (4, 4)))
    assert len(rec._buffer) == 10  # pylint: disable=protected-access


def test_trigger_is_single_flight(frame_factory):
    """A second trigger while recording is declined."""
    rec = VideoRecorder(fps=10, pre_roll_seconds=0.5, post_roll_seconds=10.0)
    for i in range(3):
        rec.add_frame(frame_factory(i, (4, 4)))
    assert rec.trigger("/tmp/a.mp4") is True
    assert rec.trigger("/tmp/b.mp4") is False


def test_encode_runs_after_post_roll(monkeypatch, tmp_path, frame_factory):
    """Once the post-roll window elapses the collected frames are encoded."""
    done = threading.Event()
    captured: dict = {}

    def _fake_encode(self, frames, dest_path):  # pylint: disable=unused-argument
        captured["frames"] = frames
        captured["dest"] = dest_path
        done.set()

    monkeypatch.setattr(VideoRecorder, "_encode", _fake_encode)

    # post_roll_seconds=0 -> the first frame after trigger completes the window.
    rec = VideoRecorder(fps=100, pre_roll_seconds=0.05, post_roll_seconds=0.0)
    rec.add_frame(frame_factory(0, (4, 4)))  # seeds the pre-roll
    dest = str(tmp_path / "clip.mp4")
    assert rec.trigger(dest) is True

    rec.add_frame(frame_factory(1, (4, 4)))  # crosses the post-roll boundary -> handoff
    assert done.wait(2.0)
    assert captured["dest"] == dest
    assert len(captured["frames"]) >= 1

    # Recording state is cleared, so a new clip can start.
    assert rec.trigger(str(tmp_path / "next.mp4")) is True


def test_add_frame_without_trigger_does_not_encode(monkeypatch, frame_factory):
    """Buffering frames while idle never kicks off an encode."""
    calls = []
    monkeypatch.setattr(
        VideoRecorder,
        "_encode",
        lambda self, frames, dest: calls.append(dest),  # pylint: disable=unused-argument
    )
    rec = VideoRecorder(fps=10, pre_roll_seconds=0.5, post_roll_seconds=1.0)
    for i in range(5):
        rec.add_frame(frame_factory(i, (4, 4)))
    assert calls == []


def test_real_encode_does_not_raise(tmp_path, frame_factory):
    """Driving the real encoder is a smoke test; asserts only if a file lands."""
    rec = VideoRecorder(fps=5, pre_roll_seconds=0.4, post_roll_seconds=0.0)
    frames = [frame_factory(i * 40, (4, 4)) for i in range(3)]
    dest = tmp_path / "clip.mp4"
    rec._encode(frames, str(dest))  # pylint: disable=protected-access
    # The OpenCV mp4v backend may be unavailable in headless CI; only assert
    # when it actually wrote the file.
    if dest.exists():
        assert dest.stat().st_size > 0
