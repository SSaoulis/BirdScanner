"""Detector control HTTP server (camera snapshots + detection deletion).

The detector owns the IMX500 camera and the data volume (database + images)
read-write, so the read-only API container cannot do these things itself.  This
module runs a tiny stdlib HTTP server on a background daemon thread inside the
detector and the API proxies browser requests through to it:

* ``GET /capture`` — capture a fresh JPEG frame (see ``backend/routers/camera.py``).
* ``DELETE /detections/{id}`` — delete a detection row + its image files (see
  ``backend/routers/detections.py``).

The port is read from the ``CAMERA_SERVER_PORT`` environment variable and
defaults to :data:`DEFAULT_CAMERA_SERVER_PORT`.
"""

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Type

import cv2
import numpy as np

logger = logging.getLogger("tracking")

DEFAULT_CAMERA_SERVER_PORT = 8000

# A callable that deletes the detection with the given id and returns True if a
# record existed (False otherwise).  See :func:`db.deleter.delete_detection`.
DeleteCallback = Callable[[int], bool]

_DETECTIONS_PREFIX = "/detections/"


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


def _parse_detection_id(path: str) -> Optional[int]:
    """Extract the integer detection id from a ``/detections/{id}`` path.

    Args:
        path: The request path, possibly including a query string.

    Returns:
        The parsed id, or ``None`` if the path does not match the expected
        ``/detections/{int}`` shape.
    """
    route = path.split("?", 1)[0]
    if not route.startswith(_DETECTIONS_PREFIX):
        return None
    tail = route[len(_DETECTIONS_PREFIX):]
    try:
        return int(tail)
    except ValueError:
        return None


def _make_handler(
    picam2: Any, delete_detection: Optional[DeleteCallback] = None
) -> Type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to a camera and delete callback.

    Args:
        picam2: The started ``Picamera2`` instance to capture frames from.
        delete_detection: Optional callable that deletes a detection by id and
            returns ``True`` if a record existed.  When ``None``, ``DELETE``
            requests are answered with 404 (deletion not configured).

    Returns:
        A ``BaseHTTPRequestHandler`` subclass serving ``GET /capture`` and
        ``DELETE /detections/{id}``.
    """

    class CameraRequestHandler(BaseHTTPRequestHandler):
        """Serves camera snapshots and detection deletions."""

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

        # do_DELETE is the stdlib-mandated handler name; the broad except is
        # intentional so any deletion failure becomes a clean 500.
        def do_DELETE(self) -> None:  # pylint: disable=invalid-name
            """Handle ``DELETE /detections/{id}``: remove a detection."""
            detection_id = _parse_detection_id(self.path)
            if delete_detection is None or detection_id is None:
                self.send_error(404, "Not Found")
                return
            try:
                deleted = delete_detection(detection_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Detection delete failed: %s", exc)
                self.send_error(500, "Delete failed")
                return
            if not deleted:
                self.send_error(404, "Detection not found")
                return
            self.send_response(204)
            self.end_headers()

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
    delete_detection: Optional[DeleteCallback] = None,
) -> ThreadingHTTPServer:
    """Start the detector control HTTP server on a background daemon thread.

    Args:
        picam2: The started ``Picamera2`` instance to capture frames from.
        port: TCP port to listen on (all interfaces).
        delete_detection: Optional callable that deletes a detection by id and
            returns ``True`` if a record existed.  When omitted, ``DELETE``
            requests are answered with 404.

    Returns:
        The running ``ThreadingHTTPServer``; call ``shutdown()`` to stop it.
    """
    server = ThreadingHTTPServer(
        ("0.0.0.0", port), _make_handler(picam2, delete_detection)
    )
    thread = threading.Thread(
        target=server.serve_forever, daemon=True, name="camera-server"
    )
    thread.start()
    logger.info("Detector control server listening on port %d", port)
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
