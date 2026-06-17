"""On-demand camera snapshot HTTP server for the detector process.

The detector owns the IMX500 camera exclusively, so the read-only API
container cannot open the camera itself.  This module runs a tiny stdlib
HTTP server on a background daemon thread inside the detector that captures
a fresh JPEG frame on demand.  The API proxies browser snapshot requests
through to it (see ``backend/routers/camera.py``).

The port is read from the ``CAMERA_SERVER_PORT`` environment variable and
defaults to :data:`DEFAULT_CAMERA_SERVER_PORT`.
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional, Type

import cv2
import numpy as np

logger = logging.getLogger("tracking")

DEFAULT_CAMERA_SERVER_PORT = 8000

# Largest crop-update request body we will read, in bytes.  The payload is a
# tiny JSON object; this guards against a malformed/huge Content-Length.
_MAX_CROP_BODY_BYTES = 4096


def encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode an RGB frame as JPEG bytes.

    The camera's ``main`` stream is configured as ``BGR888``, which (per
    picamera2's byte-reversed format naming) yields an RGB-ordered array.
    OpenCV's encoder expects BGR ordering, so the frame is converted before
    encoding to keep colours correct.

    Args:
        frame: An RGB-ordered ``numpy`` array (as produced by the camera).

    Returns:
        The encoded JPEG image as raw bytes.

    Raises:
        RuntimeError: If OpenCV fails to encode the frame.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("Failed to encode camera frame as JPEG")
    return buffer.tobytes()


def capture_jpeg(picam2: Any, stream: str = "main") -> bytes:
    """Capture a single frame from the running camera and encode it as JPEG.

    Args:
        picam2: A started ``Picamera2`` instance to capture from.
        stream: Name of the ISP output stream to capture (defaults to ``main``).

    Returns:
        The encoded JPEG image as raw bytes.

    Raises:
        RuntimeError: If OpenCV fails to encode the captured frame.
    """
    return encode_jpeg(picam2.capture_array(stream))


def _make_handler(
    picam2: Any, crop_controller: Optional[Any] = None
) -> Type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to a camera and crop controller.

    Args:
        picam2: The started ``Picamera2`` instance to capture frames from.
        crop_controller: Optional :class:`~crop_controller.CropController`
            powering the ``/crop`` and ``/capture/full`` endpoints.  When
            ``None`` those endpoints return 404 (the legacy snapshot-only mode).

    Returns:
        A ``BaseHTTPRequestHandler`` subclass serving the camera endpoints.
    """

    class CameraRequestHandler(BaseHTTPRequestHandler):
        """Serves snapshots and the detection-crop control endpoints."""

        def _send_json(self, status: int, payload: Any) -> None:
            """Serialise ``payload`` as a JSON response with ``status``."""
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_jpeg(self, jpeg: bytes) -> None:
            """Write ``jpeg`` bytes as a no-store image response."""
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)

        # do_GET is the stdlib-mandated handler name; the broad except is
        # intentional so any capture failure becomes a clean 503.
        def do_GET(self) -> None:  # pylint: disable=invalid-name
            """Route GET requests to snapshot / full-preview / crop handlers."""
            path = self.path.split("?", 1)[0]
            if path == "/capture":
                self._handle_capture()
            elif path == "/capture/full":
                self._handle_capture_full()
            elif path == "/crop":
                self._handle_get_crop()
            else:
                self.send_error(404, "Not Found")

        # do_POST is the stdlib-mandated handler name.
        def do_POST(self) -> None:  # pylint: disable=invalid-name
            """Route POST requests; only ``/crop`` is supported."""
            if self.path.split("?", 1)[0] == "/crop":
                self._handle_set_crop()
            else:
                self.send_error(404, "Not Found")

        def _handle_capture(self) -> None:
            """Capture and return a JPEG of the current (cropped) feed."""
            try:
                jpeg = capture_jpeg(picam2)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Camera capture failed: %s", exc)
                self.send_error(503, "Camera capture failed")
                return
            self._send_jpeg(jpeg)

        def _handle_capture_full(self) -> None:
            """Capture and return a JPEG of the full sensor for the crop editor."""
            if crop_controller is None:
                self.send_error(404, "Crop control not available")
                return
            try:
                jpeg = encode_jpeg(crop_controller.capture_full_preview_array())
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Full-sensor capture failed: %s", exc)
                self.send_error(503, "Camera capture failed")
                return
            self._send_jpeg(jpeg)

        def _handle_get_crop(self) -> None:
            """Return the current crop region as JSON."""
            if crop_controller is None:
                self.send_error(404, "Crop control not available")
                return
            self._send_json(200, crop_controller.get_state())

        def _handle_set_crop(self) -> None:
            """Apply a new crop from a JSON body and return the new state.

            Body is ``{"reset": true}`` to restore the default, or
            ``{"nx", "ny", "nw", "nh"}`` normalized fractions in ``[0, 1]``.
            """
            if crop_controller is None:
                self.send_error(404, "Crop control not available")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400, "Invalid Content-Length")
                return
            if length <= 0 or length > _MAX_CROP_BODY_BYTES:
                self.send_error(400, "Invalid request body size")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self.send_error(400, "Invalid JSON body")
                return
            try:
                if payload.get("reset"):
                    state = crop_controller.reset_to_default()
                else:
                    state = crop_controller.set_from_normalized(
                        float(payload["nx"]),
                        float(payload["ny"]),
                        float(payload["nw"]),
                        float(payload["nh"]),
                    )
            except (KeyError, TypeError, ValueError):
                self.send_error(400, "Body must be {reset} or {nx,ny,nw,nh}")
                return
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Failed to apply crop: %s", exc)
                self.send_error(503, "Failed to apply crop")
                return
            self._send_json(200, state)

        # Signature mirrors stdlib BaseHTTPRequestHandler.log_message.
        def log_message(  # pylint: disable=redefined-builtin
            self, format: str, *args: Any
        ) -> None:
            """Route the handler's access log through the ``tracking`` logger."""
            logger.debug("camera-server: " + format, *args)

    return CameraRequestHandler


def start_camera_server(
    picam2: Any,
    port: int = DEFAULT_CAMERA_SERVER_PORT,
    crop_controller: Optional[Any] = None,
) -> ThreadingHTTPServer:
    """Start the snapshot HTTP server on a background daemon thread.

    Args:
        picam2: The started ``Picamera2`` instance to capture frames from.
        port: TCP port to listen on (all interfaces).
        crop_controller: Optional crop controller enabling the ``/crop`` and
            ``/capture/full`` endpoints.

    Returns:
        The running ``ThreadingHTTPServer``; call ``shutdown()`` to stop it.
    """
    server = ThreadingHTTPServer(
        ("0.0.0.0", port), _make_handler(picam2, crop_controller)
    )
    thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="camera-server"
    )
    thread.start()
    logger.info("Camera snapshot server listening on port %d", port)
    return server


def camera_server_port() -> int:
    """Return the configured camera-server port from the environment.

    Reads ``CAMERA_SERVER_PORT``; falls back to
    :data:`DEFAULT_CAMERA_SERVER_PORT` when unset or not an integer.

    Returns:
        The port number to bind the snapshot server to.
    """
    raw = os.environ.get("CAMERA_SERVER_PORT")
    if raw is None:
        return DEFAULT_CAMERA_SERVER_PORT
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid CAMERA_SERVER_PORT=%r; using default %d",
            raw,
            DEFAULT_CAMERA_SERVER_PORT,
        )
        return DEFAULT_CAMERA_SERVER_PORT
