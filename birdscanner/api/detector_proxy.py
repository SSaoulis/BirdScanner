"""Shared helpers for the API routers that proxy to the detector's control server.

The API mounts the data volume read-only, so several routers forward mutating or
asset-owning requests to the detector's control server over ``httpx`` (see
``birdscanner/detector/camera_server.py``).  The one piece of logic they genuinely
share — turning the detector's ``{"error": ...}`` JSON error body into a plain
message for the relayed ``HTTPException`` — lives here so it is defined once.
"""

import httpx


def detector_error_detail(resp: httpx.Response) -> str:
    """Extract a human-readable message from a detector error response.

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
