"""Runtime-settings proxy endpoints.

The detector owns the settings overlay (it persists ``settings.json`` on the
read-write data volume and applies live changes to the running pipeline — see
``birdscanner/detector/settings_controller.py``).  The API mounts the data
volume read-only, so it cannot change settings itself; instead it proxies
browser requests through to the detector's control server (exactly like the
camera/crop routes):

* ``GET  /api/settings``          -> detector ``GET /settings``
* ``POST /api/settings``          -> detector ``POST /settings``  (partial update)
* ``POST /api/settings/restart``  -> detector ``POST /restart``

Environment variables:
    DETECTOR_URL: Base URL of the detector's control server
                  (default: ``http://detector:8000``).
"""

import os
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Body, HTTPException

router = APIRouter(prefix="/api/settings", tags=["settings"])

_DEFAULT_DETECTOR_URL = "http://detector:8000"
_TIMEOUT_SEC = 10.0


def _detector_base() -> str:
    """Return the detector control server's base URL (no trailing slash).

    Returns:
        The detector base URL from ``DETECTOR_URL`` or the default.
    """
    return os.environ.get("DETECTOR_URL", _DEFAULT_DETECTOR_URL).rstrip("/")


def _error_detail(resp: httpx.Response) -> str:
    """Extract a human-readable error message from a detector error response.

    The detector returns validation errors as ``{"error": "..."}`` JSON; fall
    back to the raw body when it is not that shape.

    Args:
        resp: The detector's error response.

    Returns:
        The extracted message, suitable for the relayed ``HTTPException`` detail.
    """
    try:
        body = resp.json()
    except ValueError:
        return resp.text
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return body["error"]
    return resp.text


@router.get("")
def get_settings() -> Dict[str, Any]:
    """Return the detector's current runtime settings + restart metadata.

    Returns:
        The settings state JSON (values + ``needs_restart`` + field metadata).

    Raises:
        HTTPException: 503 if the detector is unreachable.
    """
    try:
        resp = httpx.get(f"{_detector_base()}/settings", timeout=_TIMEOUT_SEC)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Detector unavailable: {exc}"
        ) from exc
    return resp.json()


@router.post("")
def update_settings(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Apply a partial settings update via the detector.

    Args:
        payload: A partial mapping of setting name -> new value.

    Returns:
        The updated settings state JSON.

    Raises:
        HTTPException: 503 if the detector is unreachable; relays the detector's
            4xx/5xx status (e.g. 400 for an invalid value).
    """
    try:
        resp = httpx.post(
            f"{_detector_base()}/settings", json=payload, timeout=_TIMEOUT_SEC
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Detector unavailable: {exc}"
        ) from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=_error_detail(resp))
    return resp.json()


@router.post("/restart")
def restart_detector() -> Dict[str, Any]:
    """Ask the detector to restart so restart-only settings take effect.

    Returns:
        The detector's restart-acknowledgement JSON.

    Raises:
        HTTPException: 503 if the detector is unreachable; relays the detector's
            4xx status (e.g. 404 when restart is unavailable).
    """
    try:
        resp = httpx.post(f"{_detector_base()}/restart", timeout=_TIMEOUT_SEC)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Detector unavailable: {exc}"
        ) from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()
