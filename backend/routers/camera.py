"""Camera snapshot proxy endpoint.

The detector container owns the IMX500 camera exclusively and exposes an
on-demand capture endpoint (see ``src/camera_server.py``).  The API itself has
no camera access (it mounts the data volume read-only), so this router proxies
browser snapshot requests through to the detector and relays the JPEG back.

Environment variables:
    DETECTOR_URL: Base URL of the detector's snapshot server
                  (default: ``http://detector:8000``).
"""

import os

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/api/camera", tags=["camera"])

_DEFAULT_DETECTOR_URL = "http://detector:8000"
_CAPTURE_TIMEOUT_SEC = 10.0


@router.get("/snapshot")
def get_snapshot() -> Response:
    """Proxy an on-demand camera snapshot from the detector service.

    Performs a blocking HTTP request to the detector's ``/capture`` endpoint
    (FastAPI runs this sync handler in a threadpool, so it does not block the
    event loop) and returns the JPEG it produces.

    Returns:
        A ``Response`` containing the JPEG image bytes.

    Raises:
        HTTPException: 503 if the detector is unreachable or fails to capture.
    """
    base_url = os.environ.get("DETECTOR_URL", _DEFAULT_DETECTOR_URL).rstrip("/")
    capture_url = f"{base_url}/capture"
    try:
        resp = httpx.get(capture_url, timeout=_CAPTURE_TIMEOUT_SEC)
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
