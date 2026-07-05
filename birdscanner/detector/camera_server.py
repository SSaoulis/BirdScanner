"""Detector control HTTP server (camera snapshots, crop control, deletion).

The detector owns the IMX500 camera and the data volume (database + images)
read-write, so the read-only API container cannot do these things itself.  This
module runs a tiny stdlib HTTP server on a background daemon thread inside the
detector and the API proxies browser requests through to it:

* ``GET /capture`` — capture a fresh JPEG frame (see ``birdscanner/api/routers/camera.py``).
* ``GET /capture/full`` — full-sensor JPEG for the crop editor.
* ``GET /crop`` / ``POST /crop`` — read / update the detection crop region.
* ``DELETE /detections/{id}`` — delete a detection row + its image files (see
  ``birdscanner/api/routers/detections.py``).

The port is read from the ``CAMERA_SERVER_PORT`` environment variable and
defaults to :data:`DEFAULT_CAMERA_SERVER_PORT`.

The request handler (:class:`CameraRequestHandler`) lives at module level and
reads its dependencies (the camera, crop controller, and delete callback) off
the :class:`_ControlServer` that owns it, so it stays a plain, testable class
rather than a closure built per server.
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("tracking")

DEFAULT_CAMERA_SERVER_PORT = 8000

# Largest crop-update request body we will read, in bytes.  The payload is a
# tiny JSON object; this guards against a malformed/huge Content-Length.
_MAX_CROP_BODY_BYTES = 4096

# A callable that deletes the detection with the given id and returns True if a
# record existed (False otherwise).  See :func:`db.deleter.delete_detection`.
DeleteCallback = Callable[[int], bool]

_DETECTIONS_PREFIX = "/detections/"


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
    tail = route[len(_DETECTIONS_PREFIX) :]
    try:
        return int(tail)
    except ValueError:
        return None


class _ControlServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the handler's detector dependencies.

    The dependencies live on the server (rather than in a per-server handler
    closure) so :class:`CameraRequestHandler` can be an ordinary module-level
    class that reads them via ``self.server``.
    """

    def __init__(
        self,
        server_address: tuple,
        picam2: Any,
        crop_controller: Optional[Any] = None,
        delete_detection: Optional[DeleteCallback] = None,
    ) -> None:
        """Bind the server and attach the detector dependencies.

        Args:
            server_address: The ``(host, port)`` to bind.
            picam2: The started ``Picamera2`` instance to capture frames from.
            crop_controller: Optional crop controller enabling the crop routes.
            delete_detection: Optional callable that deletes a detection by id.
        """
        super().__init__(server_address, CameraRequestHandler)
        self.picam2 = picam2
        self.crop_controller = crop_controller
        self.delete_detection = delete_detection


class CameraRequestHandler(BaseHTTPRequestHandler):
    """Serves snapshots, the crop control endpoints, and detection deletes.

    Dependencies are read from the owning :class:`_ControlServer`. With no crop
    controller the crop routes 404 (legacy snapshot-only mode); with no delete
    callback the delete route 404s.
    """

    @property
    def _control(self) -> _ControlServer:
        """Return the owning control server (with its detector dependencies)."""
        return self.server  # type: ignore[return-value]

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

    # do_DELETE is the stdlib-mandated handler name; the broad except is
    # intentional so any deletion failure becomes a clean 500.
    def do_DELETE(self) -> None:  # pylint: disable=invalid-name
        """Handle ``DELETE /detections/{id}``: remove a detection."""
        delete_detection = self._control.delete_detection
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

    def _handle_capture(self) -> None:
        """Capture and return a JPEG of the current (cropped) feed."""
        try:
            jpeg = capture_jpeg(self._control.picam2)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Camera capture failed: %s", exc)
            self.send_error(503, "Camera capture failed")
            return
        self._send_jpeg(jpeg)

    def _handle_capture_full(self) -> None:
        """Capture and return a JPEG of the full sensor for the crop editor."""
        crop_controller = self._control.crop_controller
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
        crop_controller = self._control.crop_controller
        if crop_controller is None:
            self.send_error(404, "Crop control not available")
            return
        self._send_json(200, crop_controller.get_state())

    def _read_crop_body(self) -> Optional[dict]:
        """Read + parse the crop-update JSON body, sending an error on failure.

        Returns:
            The parsed JSON object, or ``None`` when the body was missing or
            malformed (in which case an error response has already been sent).
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return None
        if length <= 0 or length > _MAX_CROP_BODY_BYTES:
            self.send_error(400, "Invalid request body size")
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "Invalid JSON body")
            return None

    def _handle_set_crop(self) -> None:
        """Apply a new crop from a JSON body and return the new state.

        Body is ``{"reset": true}`` to restore the default, or
        ``{"nx", "ny", "nw", "nh"}`` normalized fractions in ``[0, 1]``.
        """
        crop_controller = self._control.crop_controller
        if crop_controller is None:
            self.send_error(404, "Crop control not available")
            return
        payload = self._read_crop_body()
        if payload is None:
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


def start_camera_server(
    picam2: Any,
    port: int = DEFAULT_CAMERA_SERVER_PORT,
    crop_controller: Optional[Any] = None,
    delete_detection: Optional[DeleteCallback] = None,
) -> ThreadingHTTPServer:
    """Start the detector control HTTP server on a background daemon thread.

    Args:
        picam2: The started ``Picamera2`` instance to capture frames from.
        port: TCP port to listen on (all interfaces).
        crop_controller: Optional crop controller enabling the ``/crop`` and
            ``/capture/full`` endpoints.
        delete_detection: Optional callable that deletes a detection by id and
            returns ``True`` if a record existed.  When omitted, ``DELETE``
            requests are answered with 404.

    Returns:
        The running ``ThreadingHTTPServer``; call ``shutdown()`` to stop it.
    """
    server = _ControlServer(
        ("0.0.0.0", port), picam2, crop_controller, delete_detection
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
