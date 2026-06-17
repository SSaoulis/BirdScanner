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
