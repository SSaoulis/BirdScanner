"""Unit tests for VideoRecorder buffer + trigger logic.

Frames come from the shared ``frame_factory`` fixture. The actual mp4 encode
depends on the OpenCV FFmpeg backend, which is not guaranteed in every CI
environment, so the handoff-to-encode is exercised by monkeypatching ``_encode``.
One best-effort test drives the real encoder but only asserts on the file when the
backend produced one.
"""

import threading

from birdscanner.detector import video_recorder as vr
from birdscanner.detector.video_recorder import VideoRecorder, _open_writer


class _FakeWriter:
    """Stand-in for ``cv2.VideoWriter`` that reports a preset ``isOpened``."""

    def __init__(self, opened):
        self._opened = opened
        self.released = False

    def isOpened(self):  # pylint: disable=invalid-name  # mirrors cv2's method
        """Return the preset open state."""
        return self._opened

    def release(self):
        """Record that the writer was released."""
        self.released = True


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
        lambda self, frames, dest: calls.append(
            dest
        ),  # pylint: disable=unused-argument
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


def test_encode_with_no_frames_is_a_noop(tmp_path):
    """Encoding an empty frame list writes nothing and returns without raising."""
    rec = VideoRecorder(fps=5, pre_roll_seconds=0.4, post_roll_seconds=0.0)
    dest = tmp_path / "empty.mp4"
    rec._encode([], str(dest))  # pylint: disable=protected-access
    assert not dest.exists()


def test_open_writer_prefers_h264(monkeypatch):
    """avc1 is tried first and used when it opens; mp4v is never reached."""
    codecs = []

    def _factory(_dest, fourcc, _fps, _size):
        codecs.append(fourcc)
        return _FakeWriter(opened=True)  # first codec (avc1) opens

    monkeypatch.setattr(vr.cv2, "VideoWriter", _factory)
    monkeypatch.setattr(vr.cv2, "VideoWriter_fourcc", lambda *chars: "".join(chars))

    writer = _open_writer("/tmp/clip.mp4", 10.0, (4, 4))
    assert writer is not None
    assert codecs == ["avc1"]  # stopped at the first working codec


def test_open_writer_falls_back_to_mp4v(monkeypatch):
    """When avc1 will not open, the helper falls through to mp4v."""
    codecs = []

    def _factory(_dest, fourcc, _fps, _size):
        codecs.append(fourcc)
        return _FakeWriter(opened=(fourcc == "mp4v"))

    monkeypatch.setattr(vr.cv2, "VideoWriter", _factory)
    monkeypatch.setattr(vr.cv2, "VideoWriter_fourcc", lambda *chars: "".join(chars))

    writer = _open_writer("/tmp/clip.mp4", 10.0, (4, 4))
    assert writer is not None
    assert writer.isOpened()
    assert codecs == ["avc1", "mp4v"]  # avc1 tried and rejected, mp4v accepted


def test_open_writer_returns_none_when_no_codec_opens(monkeypatch):
    """No working codec yields None (and every attempted writer is released)."""
    attempted = []

    def _factory(_dest, fourcc, _fps, _size):
        writer = _FakeWriter(opened=False)
        attempted.append(writer)
        return writer

    monkeypatch.setattr(vr.cv2, "VideoWriter", _factory)
    monkeypatch.setattr(vr.cv2, "VideoWriter_fourcc", lambda *chars: "".join(chars))

    assert _open_writer("/tmp/clip.mp4", 10.0, (4, 4)) is None
    assert all(w.released for w in attempted)  # no leaked writers


def test_encode_returns_when_no_codec_opens(monkeypatch, tmp_path, frame_factory):
    """_encode logs and returns (no crash) when no writer can be opened."""
    monkeypatch.setattr(vr, "_open_writer", lambda *args, **kwargs: None)
    rec = VideoRecorder(fps=5, pre_roll_seconds=0.4, post_roll_seconds=0.0)
    dest = tmp_path / "clip.mp4"
    rec._encode(  # pylint: disable=protected-access
        [frame_factory(0, (4, 4))], str(dest)
    )
    assert not dest.exists()
