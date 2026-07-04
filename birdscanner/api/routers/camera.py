"""Camera snapshot + detection-crop proxy endpoints.

The detector container owns the IMX500 camera exclusively and exposes on-demand
capture and crop-control endpoints (see ``birdscanner/detector/camera_server.py``).  The API
itself has no camera access (it mounts the data volume read-only), so this
router proxies browser requests through to the detector and relays the result
back:

* ``GET  /api/camera/snapshot``       -> detector ``GET /capture``       (JPEG)
* ``GET  /api/camera/snapshot/full``  -> detector ``GET /capture/full``  (JPEG)
* ``GET  /api/camera/crop``           -> detector ``GET /crop``          (JSON)
* ``POST /api/camera/crop``           -> detector ``POST /crop``         (JSON)

Environment variables:
    DETECTOR_URL: Base URL of the detector's snapshot server
                  (default: ``http://detector:8000``).
"""

import os
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/api/camera", tags=["camera"])

_DEFAULT_DETECTOR_URL = "http://detector:8000"
_CAPTURE_TIMEOUT_SEC = 10.0
# The full-sensor preview briefly widens the crop and waits for it to settle, so
# it can take longer than a plain snapshot.
_FULL_CAPTURE_TIMEOUT_SEC = 15.0


def _detector_base() -> str:
    """Return the detector snapshot server's base URL (no trailing slash).

    Returns:
        The detector base URL from ``DETECTOR_URL`` or the default.
    """
    return os.environ.get("DETECTOR_URL", _DEFAULT_DETECTOR_URL).rstrip("/")


def _proxy_jpeg(path: str, timeout: float) -> Response:
    """Proxy a JPEG-producing detector endpoint and relay the image.

    Args:
        path: Detector path to request (e.g. ``/capture``).
        timeout: Request timeout in seconds.

    Returns:
        A ``Response`` containing the relayed JPEG bytes.

    Raises:
        HTTPException: 503 if the detector is unreachable or fails to capture.
    """
    try:
        resp = httpx.get(f"{_detector_base()}{path}", timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Camera unavailable: {exc}"
        ) from exc
    media_type = resp.headers.get("Content-Type", "image/jpeg")
    return Response(
        content=resp.content,
        media_type=media_type,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/snapshot")
def get_snapshot() -> Response:
    """Proxy an on-demand snapshot of the current (cropped) detection feed.

    Returns:
        A ``Response`` containing the JPEG image bytes.

    Raises:
        HTTPException: 503 if the detector is unreachable or fails to capture.
    """
    return _proxy_jpeg("/capture", _CAPTURE_TIMEOUT_SEC)


@router.get("/snapshot/full")
def get_full_snapshot() -> Response:
    """Proxy a full-sensor snapshot for the crop editor.

    The detector momentarily widens the crop to the whole sensor to produce this
    frame, so the user can see the entire scene while repositioning the box.

    Returns:
        A ``Response`` containing the JPEG image bytes.

    Raises:
        HTTPException: 503 if the detector is unreachable or fails to capture.
    """
    return _proxy_jpeg("/capture/full", _FULL_CAPTURE_TIMEOUT_SEC)


@router.get("/crop")
def get_crop() -> Dict[str, Any]:
    """Return the detector's current detection-crop region.

    Returns:
        The crop state JSON (sensor pixels, normalized box, sensor dimensions).

    Raises:
        HTTPException: 503 if the detector is unreachable.
    """
    try:
        resp = httpx.get(f"{_detector_base()}/crop", timeout=_CAPTURE_TIMEOUT_SEC)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Camera unavailable: {exc}"
        ) from exc
    return resp.json()


@router.post("/crop")
def set_crop(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Apply a new detection-crop region via the detector.

    Args:
        payload: Either ``{"reset": true}`` or a normalized box
            ``{"nx", "ny", "nw", "nh"}`` (fractions in ``[0, 1]``).

    Returns:
        The updated crop state JSON.

    Raises:
        HTTPException: 503 if the detector is unreachable; relays the detector's
            4xx status (e.g. 400 for an invalid body).
    """
    try:
        resp = httpx.post(
            f"{_detector_base()}/crop", json=payload, timeout=_CAPTURE_TIMEOUT_SEC
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Camera unavailable: {exc}"
        ) from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()
