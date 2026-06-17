"""Tests for the detector's on-demand camera snapshot server.

These exercise the pure JPEG-encoding helper and the stdlib HTTP server with a
fake ``Picamera2`` stand-in, so no real camera (or picamera2 install) is needed.
"""

import urllib.error
import urllib.request

import numpy as np
import pytest

from src.camera_server import (
    DEFAULT_CAMERA_SERVER_PORT,
    camera_server_port,
    capture_jpeg,
    start_camera_server,
)

_JPEG_MAGIC = b"\xff\xd8\xff"


class _FakeCamera:
    """Minimal ``Picamera2`` stand-in returning a fixed frame."""

    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.requested_stream: str | None = None

    def capture_array(self, stream: str) -> np.ndarray:
        self.requested_stream = stream
        return self.frame


class _BrokenCamera:
    """Camera stand-in whose capture always fails."""

    def capture_array(self, stream: str) -> np.ndarray:
        raise RuntimeError("camera not ready")


def _solid_frame() -> np.ndarray:
    """A 16x16 RGB frame of a uniform colour."""
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    return frame


def test_capture_jpeg_returns_jpeg_bytes() -> None:
    cam = _FakeCamera(_solid_frame())
    jpeg = capture_jpeg(cam)
    assert isinstance(jpeg, bytes)
    assert jpeg.startswith(_JPEG_MAGIC)
    assert cam.requested_stream == "main"


def test_camera_server_port_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAMERA_SERVER_PORT", raising=False)
    assert camera_server_port() == DEFAULT_CAMERA_SERVER_PORT


def test_camera_server_port_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAMERA_SERVER_PORT", "9123")
    assert camera_server_port() == 9123


def test_camera_server_port_invalid_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMERA_SERVER_PORT", "not-a-port")
    assert camera_server_port() == DEFAULT_CAMERA_SERVER_PORT


@pytest.fixture()
def running_server(request):
    """Start the snapshot server on an ephemeral port; yield (base_url, camera)."""
    camera = request.param
    server = start_camera_server(camera, port=0)
    host, port = server.server_address
    try:
        yield f"http://127.0.0.1:{port}", camera
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("running_server", [_FakeCamera(_solid_frame())], indirect=True)
def test_capture_endpoint_serves_jpeg(running_server) -> None:
    base_url, _ = running_server
    with urllib.request.urlopen(f"{base_url}/capture", timeout=5) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "image/jpeg"
        body = resp.read()
    assert body.startswith(_JPEG_MAGIC)


@pytest.mark.parametrize("running_server", [_FakeCamera(_solid_frame())], indirect=True)
def test_unknown_path_returns_404(running_server) -> None:
    base_url, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/nope", timeout=5)
    assert exc_info.value.code == 404


@pytest.mark.parametrize("running_server", [_BrokenCamera()], indirect=True)
def test_capture_failure_returns_503(running_server) -> None:
    base_url, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/capture", timeout=5)
    assert exc_info.value.code == 503


# ---------------------------------------------------------------------------
# Detection deletion (DELETE /detections/{id})
# ---------------------------------------------------------------------------


def _delete(url: str) -> int:
    """Issue a DELETE request and return the HTTP status code."""
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status


@pytest.fixture()
def deleting_server(request):
    """Start the control server with a delete callback; yield (base_url, calls).

    ``request.param`` is the callback's return value (True/False) or an
    exception instance to raise.  ``calls`` records the ids passed in.
    """
    outcome = request.param
    calls: list[int] = []

    def _delete_callback(detection_id: int) -> bool:
        calls.append(detection_id)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    server = start_camera_server(
        _FakeCamera(_solid_frame()), port=0, delete_detection=_delete_callback
    )
    host, port = server.server_address
    try:
        yield f"http://127.0.0.1:{port}", calls
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("deleting_server", [True], indirect=True)
def test_delete_existing_returns_204(deleting_server) -> None:
    base_url, calls = deleting_server
    assert _delete(f"{base_url}/detections/42") == 204
    assert calls == [42]


@pytest.mark.parametrize("deleting_server", [False], indirect=True)
def test_delete_missing_returns_404(deleting_server) -> None:
    base_url, calls = deleting_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(
            urllib.request.Request(f"{base_url}/detections/7", method="DELETE"),
            timeout=5,
        )
    assert exc_info.value.code == 404
    assert calls == [7]


@pytest.mark.parametrize("deleting_server", [True], indirect=True)
def test_delete_non_integer_id_returns_404(deleting_server) -> None:
    base_url, calls = deleting_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(
            urllib.request.Request(f"{base_url}/detections/abc", method="DELETE"),
            timeout=5,
        )
    assert exc_info.value.code == 404
    assert calls == []


@pytest.mark.parametrize("deleting_server", [RuntimeError("boom")], indirect=True)
def test_delete_callback_error_returns_500(deleting_server) -> None:
    base_url, _ = deleting_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(
            urllib.request.Request(f"{base_url}/detections/1", method="DELETE"),
            timeout=5,
        )
    assert exc_info.value.code == 500


@pytest.mark.parametrize("running_server", [_FakeCamera(_solid_frame())], indirect=True)
def test_delete_without_callback_returns_404(running_server) -> None:
    base_url, _ = running_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(
            urllib.request.Request(f"{base_url}/detections/1", method="DELETE"),
            timeout=5,
        )
    assert exc_info.value.code == 404
