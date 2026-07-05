"""Tests for the detector's on-demand camera snapshot server.

These exercise the pure JPEG-encoding helper and the stdlib HTTP server with a
fake ``Picamera2`` stand-in, so no real camera (or picamera2 install) is needed.
"""

import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from birdscanner.detector.camera_server import (
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
    _, port = server.server_address
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
    _, port = server.server_address
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


# ---------------------------------------------------------------------------
# Crop control endpoints (/crop, /capture/full)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("running_server", [_FakeCamera(_solid_frame())], indirect=True)
def test_crop_endpoints_absent_without_controller(running_server) -> None:
    """Without a crop controller, the crop endpoints 404 (legacy mode)."""
    base_url, _ = running_server
    for path in ("/crop", "/capture/full"):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base_url}{path}", timeout=5)
        assert exc_info.value.code == 404


class _FakeCropController:
    """Crop-controller stand-in recording the calls the server makes."""

    def __init__(self) -> None:
        self.last_set = None
        self.reset_called = False
        self._state = {
            "x": 100,
            "y": 200,
            "w": 900,
            "h": 900,
            "norm": {"nx": 0.1, "ny": 0.2, "nw": 0.3, "nh": 0.4},
            "sensor_w": 4056,
            "sensor_h": 3040,
        }

    def get_state(self):
        return self._state

    def set_from_normalized(self, nx, ny, nw, nh):
        self.last_set = (nx, ny, nw, nh)
        return self._state

    def reset_to_default(self):
        self.reset_called = True
        return self._state

    def capture_full_preview_array(self):
        return _solid_frame()


@pytest.fixture()
def crop_server():
    """Start the server with a fake crop controller; yield (base_url, controller)."""
    controller = _FakeCropController()
    server = start_camera_server(
        _FakeCamera(_solid_frame()), port=0, crop_controller=controller
    )
    _, port = server.server_address
    try:
        yield f"http://127.0.0.1:{port}", controller
    finally:
        server.shutdown()
        server.server_close()


def test_get_crop_returns_state(crop_server) -> None:
    base_url, _ = crop_server
    with urllib.request.urlopen(f"{base_url}/crop", timeout=5) as resp:
        assert resp.status == 200
        body = json.loads(resp.read())
    assert body["w"] == 900
    assert body["norm"]["nx"] == 0.1
    assert body["sensor_w"] == 4056


def test_capture_full_serves_jpeg(crop_server) -> None:
    base_url, _ = crop_server
    with urllib.request.urlopen(f"{base_url}/capture/full", timeout=5) as resp:
        assert resp.status == 200
        body = resp.read()
    assert body.startswith(_JPEG_MAGIC)


def _post_json(url: str, payload: dict):
    """POST a JSON body and return the parsed response (raises on HTTP error)."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


def test_post_crop_normalized_box(crop_server) -> None:
    base_url, controller = crop_server
    status, body = _post_json(
        f"{base_url}/crop", {"nx": 0.1, "ny": 0.2, "nw": 0.3, "nh": 0.4}
    )
    assert status == 200
    assert body["w"] == 900
    assert controller.last_set == (0.1, 0.2, 0.3, 0.4)


def test_post_crop_reset(crop_server) -> None:
    base_url, controller = crop_server
    status, _ = _post_json(f"{base_url}/crop", {"reset": True})
    assert status == 200
    assert controller.reset_called is True


def test_post_crop_missing_keys_returns_400(crop_server) -> None:
    base_url, _ = crop_server
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _post_json(f"{base_url}/crop", {"nx": 0.1})
    assert exc_info.value.code == 400


def test_post_crop_invalid_json_returns_400(crop_server) -> None:
    base_url, _ = crop_server
    request = urllib.request.Request(
        f"{base_url}/crop",
        data=b"not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request, timeout=5)
    assert exc_info.value.code == 400
