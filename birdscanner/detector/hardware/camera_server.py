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
* ``PATCH /detections/{id}`` — correct a detection's species (moves its files to
  the corrected species folder; see ``birdscanner/db/corrector.py``).
* ``GET /labels`` — the species vocabulary (classifier classes plus user-added
  custom labels) for the correction picker.

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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("tracking")

DEFAULT_CAMERA_SERVER_PORT = 8000

# Largest request body we will read, in bytes.  The crop/settings payloads are
# tiny JSON objects; this guards against a malformed/huge Content-Length.
_MAX_BODY_BYTES = 4096

# A callable that deletes the detection with the given id and returns True if a
# record existed (False otherwise).  See :func:`db.deleter.delete_detection`.
DeleteCallback = Callable[[int], bool]

# A callable that reassigns a detection's species and returns the updated record
# as a JSON-serialisable dict (or None if no such detection exists).  See
# :func:`db.corrector.correct_detection_species`.
CorrectCallback = Callable[[int, str], Optional[dict]]

# A callable returning the current custom (user-added) species labels, so the
# served vocabulary reflects labels added while the detector runs.  See
# :func:`db.custom_species.list_custom_species`.
CustomSpeciesProvider = Callable[[], list[str]]

# A callable that registers a new custom species label and returns the canonical
# stored name.  See :func:`db.custom_species.add_custom_species`.
RegisterSpeciesCallback = Callable[[str], str]

# A callable that asks the detector to restart (typically by scheduling process
# exit so Docker's restart policy relaunches it).
RestartCallback = Callable[[], None]

_DETECTIONS_PREFIX = "/detections/"

# Upper bound on a user-added species label's length, guarding against a runaway
# free-text field creating an absurd folder/label.
_MAX_SPECIES_NAME_LEN = 100


@dataclass
class ControlServerDeps:
    """Optional detector dependencies enabling the control server's routes.

    Bundled into one value object (rather than passed as separate arguments) so
    :class:`_ControlServer` / :func:`start_camera_server` keep short signatures.
    Each dependency independently enables its routes; when one is ``None`` the
    corresponding routes answer 404 (legacy snapshot-only mode).

    Attributes:
        crop_controller: Enables the ``/crop`` and ``/capture/full`` routes.
        delete_detection: Enables ``DELETE /detections/{id}``.
        settings_controller: Enables ``GET``/``POST /settings``.
        restart: Enables ``POST /restart``.
        correct_species: Enables ``PATCH /detections/{id}`` (species correction).
        species_labels: The classifier vocabulary; enables ``GET /labels`` and
            gates species corrections to a known label.
        list_custom_species: Returns the user-added species labels, unioned with
            ``species_labels`` for ``GET /labels`` and correction validation so
            labels added at runtime are recognised immediately.
        register_species: Persists a new user-added species label (returning its
            canonical form) — enables the ``allow_new`` branch of a correction so a
            bird outside the classifier's vocabulary can be recorded.
    """

    crop_controller: Optional[Any] = None
    delete_detection: Optional[DeleteCallback] = None
    settings_controller: Optional[Any] = None
    restart: Optional[RestartCallback] = None
    correct_species: Optional[CorrectCallback] = None
    species_labels: Optional[list[str]] = None
    list_custom_species: Optional[CustomSpeciesProvider] = None
    register_species: Optional[RegisterSpeciesCallback] = None


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


def _canonical_known(name: str, known: set[str]) -> Optional[str]:
    """Return the known label matching ``name`` case-insensitively, else ``None``.

    Args:
        name: A candidate species label (leading/trailing space is ignored).
        known: The set of known species labels.

    Returns:
        The existing known label whose lower-cased form equals ``name``'s (so a
        differently-cased spelling resolves to the canonical one), or ``None`` when
        no such label exists.
    """
    lowered = name.strip().lower()
    for label in known:
        if label.lower() == lowered:
            return label
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
        deps: ControlServerDeps,
    ) -> None:
        """Bind the server and attach the detector dependencies.

        Args:
            server_address: The ``(host, port)`` to bind.
            picam2: The started ``Picamera2`` instance to capture frames from.
            deps: Optional detector dependencies enabling the extra routes.
        """
        super().__init__(server_address, CameraRequestHandler)
        self.picam2 = picam2
        self.deps = deps


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
        """Route GET requests to snapshot / full-preview / crop / settings."""
        path = self.path.split("?", 1)[0]
        if path == "/capture":
            self._handle_capture()
        elif path == "/capture/full":
            self._handle_capture_full()
        elif path == "/crop":
            self._handle_get_crop()
        elif path == "/settings":
            self._handle_get_settings()
        elif path == "/labels":
            self._handle_get_labels()
        else:
            self.send_error(404, "Not Found")

    # do_POST is the stdlib-mandated handler name.
    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Route POST requests to the crop / settings / restart handlers."""
        path = self.path.split("?", 1)[0]
        if path == "/crop":
            self._handle_set_crop()
        elif path == "/settings":
            self._handle_set_settings()
        elif path == "/restart":
            self._handle_restart()
        else:
            self.send_error(404, "Not Found")

    # do_DELETE is the stdlib-mandated handler name; the broad except is
    # intentional so any deletion failure becomes a clean 500.
    def do_DELETE(self) -> None:  # pylint: disable=invalid-name
        """Handle ``DELETE /detections/{id}``: remove a detection."""
        delete_detection = self._control.deps.delete_detection
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

    # do_PATCH is the stdlib-mandated handler name; the broad except turns any
    # correction failure into a clean 500.
    def do_PATCH(self) -> None:  # pylint: disable=invalid-name
        """Handle ``PATCH /detections/{id}``: correct a detection's species."""
        self._handle_correct_species()

    def _handle_correct_species(self) -> None:
        """Reassign a detection's species from a JSON body ``{"species": ...}``.

        The species must be a non-empty string. It is accepted when it is already a
        known label (classifier vocabulary or a previously-added custom label);
        otherwise it is rejected with a 400 JSON ``{"error": ...}`` (the shape that
        survives the API proxy) — unless the body sets ``allow_new`` truthy, in
        which case the label is registered as a new custom species and used. Returns
        the updated record as JSON on success, or 404 when the detection does not
        exist.
        """
        correct_species = self._control.deps.correct_species
        detection_id = _parse_detection_id(self.path)
        if correct_species is None or detection_id is None:
            self.send_error(404, "Not Found")
            return
        payload = self._read_json_body()
        if payload is None:
            return
        species = payload.get("species")
        if not isinstance(species, str) or not species.strip():
            self._send_json(400, {"error": "Missing or invalid 'species'"})
            return
        known = self._known_species()
        if known is not None and species not in known:
            resolved = self._resolve_new_species(
                species, known, bool(payload.get("allow_new"))
            )
            if resolved is None:
                return  # an error response has already been sent
            species = resolved
        try:
            record = correct_species(detection_id, species)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Species correction failed: %s", exc)
            self.send_error(500, "Correction failed")
            return
        if record is None:
            self.send_error(404, "Detection not found")
            return
        self._send_json(200, record)

    def _known_species(self) -> Optional[set[str]]:
        """Return the set of known species labels, or ``None`` when ungated.

        The set is the union of the classifier vocabulary (``species_labels``) and
        the current custom labels (``list_custom_species``). When neither dependency
        is configured, returns ``None`` to signal that corrections are not gated
        (legacy snapshot-only mode), preserving the pre-existing behaviour.
        """
        labels = self._control.deps.species_labels
        provider = self._control.deps.list_custom_species
        if labels is None and provider is None:
            return None
        known: set[str] = set(labels or [])
        if provider is not None:
            known |= set(provider())
        return known

    def _resolve_new_species(
        self, species: str, known: set[str], allow_new: bool
    ) -> Optional[str]:
        """Resolve a species not exactly present in ``known`` to a label to store.

        A case-insensitive match against an existing known label is reused (so a
        differently-cased spelling never creates a near-duplicate). Otherwise the
        label is only accepted when ``allow_new`` is set and a
        :attr:`ControlServerDeps.register_species` callback is available, in which
        case it is registered as a new custom species and its canonical form
        returned.

        Args:
            species: The (non-empty) requested species label.
            known: The current set of known labels.
            allow_new: Whether the request opted in to creating a new label.

        Returns:
            The canonical label to correct with, or ``None`` when the label was
            rejected (in which case an error response has already been sent).
        """
        canonical = _canonical_known(species, known)
        if canonical is not None:
            return canonical
        register_species = self._control.deps.register_species
        if not allow_new or register_species is None:
            self._send_json(400, {"error": f"Unknown species '{species}'"})
            return None
        candidate = species.strip()
        if len(candidate) > _MAX_SPECIES_NAME_LEN:
            self._send_json(400, {"error": "Species name too long"})
            return None
        try:
            return register_species(candidate)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return None

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
        crop_controller = self._control.deps.crop_controller
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
        crop_controller = self._control.deps.crop_controller
        if crop_controller is None:
            self.send_error(404, "Crop control not available")
            return
        self._send_json(200, crop_controller.get_state())

    def _read_json_body(self) -> Optional[dict]:
        """Read + parse a small JSON request body, sending an error on failure.

        Returns:
            The parsed JSON object, or ``None`` when the body was missing or
            malformed (in which case an error response has already been sent).
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return None
        if length <= 0 or length > _MAX_BODY_BYTES:
            self.send_error(400, "Invalid request body size")
            return None
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_error(400, "Invalid JSON body")
            return None
        if not isinstance(body, dict):
            self.send_error(400, "Body must be a JSON object")
            return None
        return body

    def _handle_set_crop(self) -> None:
        """Apply a new crop from a JSON body and return the new state.

        Body is ``{"reset": true}`` to restore the default, or
        ``{"nx", "ny", "nw", "nh"}`` normalized fractions in ``[0, 1]``.
        """
        crop_controller = self._control.deps.crop_controller
        if crop_controller is None:
            self.send_error(404, "Crop control not available")
            return
        payload = self._read_json_body()
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

    def _handle_get_labels(self) -> None:
        """Return the species vocabulary as ``{"species": [...]}``.

        The list is the sorted union of the classifier vocabulary and the
        user-added custom labels, so a label added at runtime appears in the picker
        immediately.
        """
        known = self._known_species()
        if known is None:
            self.send_error(404, "Labels not available")
            return
        self._send_json(200, {"species": sorted(known)})

    def _handle_get_settings(self) -> None:
        """Return the current runtime settings + restart metadata as JSON."""
        settings_controller = self._control.deps.settings_controller
        if settings_controller is None:
            self.send_error(404, "Settings control not available")
            return
        self._send_json(200, settings_controller.get_state())

    def _handle_set_settings(self) -> None:
        """Apply a partial settings update from a JSON body; return new state.

        A validation error (unknown key / out-of-range value) is answered 400
        with the message; a persistence failure is answered 500.
        """
        settings_controller = self._control.deps.settings_controller
        if settings_controller is None:
            self.send_error(404, "Settings control not available")
            return
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            state = settings_controller.update(payload)
        except ValueError as exc:
            # JSON (not send_error's HTML) so the message survives the API proxy
            # and reaches the Settings page.
            self._send_json(400, {"error": str(exc)})
            return
        except OSError as exc:
            logger.warning("Failed to persist settings: %s", exc)
            self._send_json(500, {"error": "Failed to save settings"})
            return
        self._send_json(200, state)

    def _handle_restart(self) -> None:
        """Ask the detector to restart, returning 202 once scheduled."""
        restart = self._control.deps.restart
        if restart is None:
            self.send_error(404, "Restart not available")
            return
        restart()
        self._send_json(202, {"status": "restarting"})

    # Signature mirrors stdlib BaseHTTPRequestHandler.log_message.
    def log_message(  # pylint: disable=redefined-builtin
        self, format: str, *args: Any
    ) -> None:
        """Route the handler's access log through the ``tracking`` logger."""
        logger.debug("camera-server: " + format, *args)


def start_camera_server(
    picam2: Any,
    port: int = DEFAULT_CAMERA_SERVER_PORT,
    deps: Optional[ControlServerDeps] = None,
) -> ThreadingHTTPServer:
    """Start the detector control HTTP server on a background daemon thread.

    Args:
        picam2: The started ``Picamera2`` instance to capture frames from.
        port: TCP port to listen on (all interfaces).
        deps: Optional detector dependencies enabling the crop, delete, settings,
            and restart routes.  Routes whose dependency is absent answer 404.

    Returns:
        The running ``ThreadingHTTPServer``; call ``shutdown()`` to stop it.
    """
    server = _ControlServer(("0.0.0.0", port), picam2, deps or ControlServerDeps())
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
