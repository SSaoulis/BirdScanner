"""On-demand camera snapshot HTTP server for the detector process.

The detector owns the IMX500 camera exclusively, so the read-only API
container cannot open the camera itself.  This module runs a tiny stdlib
HTTP server on a background daemon thread inside the detector that captures
a fresh JPEG frame on demand.  The API proxies browser snapshot requests
through to it (see ``backend/routers/camera.py``).

The port is read from the ``CAMERA_SERVER_PORT`` environment variable and
defaults to :data:`DEFAULT_CAMERA_SERVER_PORT`.
"""

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Type

import cv2
import numpy as np

logger = logging.getLogger("tracking")

DEFAULT_CAMERA_SERVER_PORT = 8000


def capture_jpeg(picam2: Any, stream: str = "main") -> bytes:
    """Capture a single frame from the running camera and encode it as JPEG.

    The camera's ``main`` stream is configured as ``BGR888``, which (per
    picamera2's byte-reversed format naming) yields an RGB-ordered array.
    OpenCV's encoder expects BGR ordering, so the frame is converted before
    encoding to keep colours correct.

    Args:
        picam2: A started ``Picamera2`` instance to capture from.
        stream: Name of the ISP output stream to capture (defaults to ``main``).

    Returns:
        The encoded JPEG image as raw bytes.

    Raises:
        RuntimeError: If OpenCV fails to encode the captured frame.
    """
    frame: np.ndarray = picam2.capture_array(stream)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("Failed to encode camera frame as JPEG")
    return buffer.tobytes()


def _make_handler(picam2: Any) -> Type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to a specific camera instance.

    Args:
        picam2: The started ``Picamera2`` instance to capture frames from.

    Returns:
        A ``BaseHTTPRequestHandler`` subclass that serves ``GET /capture``.
    """

    class CameraRequestHandler(BaseHTTPRequestHandler):
        """Serves a freshly captured JPEG frame on ``GET /capture``."""

        # do_GET is the stdlib-mandated handler name; the broad except is
        # intentional so any capture failure becomes a clean 503.
        def do_GET(self) -> None:  # pylint: disable=invalid-name
            """Handle a GET request, capturing and returning a JPEG frame."""
            if self.path.split("?", 1)[0] != "/capture":
                self.send_error(404, "Not Found")
                return
            try:
                jpeg = capture_jpeg(picam2)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Camera capture failed: %s", exc)
                self.send_error(503, "Camera capture failed")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)

        # Signature mirrors stdlib BaseHTTPRequestHandler.log_message.
        def log_message(  # pylint: disable=redefined-builtin
            self, format: str, *args: Any
        ) -> None:
            """Route the handler's access log through the ``tracking`` logger."""
            logger.debug("camera-server: " + format, *args)

    return CameraRequestHandler


def start_camera_server(
    picam2: Any, port: int = DEFAULT_CAMERA_SERVER_PORT
) -> ThreadingHTTPServer:
    """Start the snapshot HTTP server on a background daemon thread.

    Args:
        picam2: The started ``Picamera2`` instance to capture frames from.
        port: TCP port to listen on (all interfaces).

    Returns:
        The running ``ThreadingHTTPServer``; call ``shutdown()`` to stop it.
    """
    server = ThreadingHTTPServer(("0.0.0.0", port), _make_handler(picam2))
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
