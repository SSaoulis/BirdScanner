"""Detection list, detail, and delete endpoints."""

import os
from datetime import datetime
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlmodel import Session, select

from db.models import DetectionRecord
from backend.dependencies import get_session

router = APIRouter(prefix="/api/detections", tags=["detections"])

# The API mounts the database and images read-only, so it cannot delete them
# itself.  Deletes are proxied to the detector's control server (it owns the
# data volume read-write), mirroring the camera-snapshot proxy.
_DEFAULT_DETECTOR_URL = "http://detector:8000"
_DELETE_TIMEOUT_SEC = 10.0


@router.get("", response_model=List[DetectionRecord])
def list_detections(
    species: Optional[str] = Query(default=None, description="Filter by species name"),
    from_: Optional[datetime] = Query(
        default=None, alias="from", description="Earliest timestamp (inclusive)"
    ),
    to: Optional[datetime] = Query(
        default=None, description="Latest timestamp (inclusive)"
    ),
    min_confidence: Optional[float] = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Only return detections with confidence at or above this value (0–1)",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    session: Session = Depends(get_session),
) -> List[DetectionRecord]:
    """Return a paginated, optionally filtered list of detection records.

    Args:
        species: If provided, only return detections for this species.
        from_: If provided, only return detections at or after this timestamp.
        to: If provided, only return detections at or before this timestamp.
        min_confidence: If provided, only return detections whose confidence is
            at or above this value (0–1).
        limit: Maximum number of records to return (1–500, default 50).
        offset: Number of records to skip for pagination.
        session: Injected database session.

    Returns:
        List of ``DetectionRecord`` objects ordered by timestamp descending,
        with ``id`` descending as a tiebreaker for a deterministic page order.
    """
    query = select(DetectionRecord)
    if species is not None:
        query = query.where(DetectionRecord.species == species)
    if from_ is not None:
        query = query.where(DetectionRecord.timestamp >= from_)
    if to is not None:
        query = query.where(DetectionRecord.timestamp <= to)
    if min_confidence is not None:
        query = query.where(DetectionRecord.confidence >= min_confidence)
    # Order by timestamp, then id, so rows with identical timestamps keep a
    # stable order across paginated requests. Without the id tiebreaker SQLite
    # may return tied rows in a different order per query, which makes
    # offset-based pages overlap and surfaces duplicate detections in the UI.
    query = (
        query.order_by(
            DetectionRecord.timestamp.desc(),  # type: ignore[attr-defined]  # pylint: disable=no-member
            DetectionRecord.id.desc(),  # type: ignore[attr-defined]  # pylint: disable=no-member
        )
        .offset(offset)
        .limit(limit)
    )
    return list(session.exec(query).all())


@router.get("/{detection_id}", response_model=DetectionRecord)
def get_detection(
    detection_id: int,
    session: Session = Depends(get_session),
) -> DetectionRecord:
    """Return a single detection record by ID.

    Args:
        detection_id: Primary key of the detection.
        session: Injected database session.

    Returns:
        The matching ``DetectionRecord``.

    Raises:
        HTTPException: 404 if no detection with that ID exists.
    """
    record = session.get(DetectionRecord, detection_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    return record


@router.delete("/{detection_id}", status_code=204)
def delete_detection(detection_id: int) -> Response:
    """Delete a detection by proxying to the detector's control server.

    The API has no write access to the database or image files; the detector
    owns them (see ``src/camera_server.py``).  This endpoint forwards the
    delete to the detector and relays the outcome.

    Args:
        detection_id: Primary key of the detection to delete.

    Returns:
        An empty ``204 No Content`` response on success.

    Raises:
        HTTPException: 404 if the detection does not exist, 503 if the detector
            is unreachable, or 502 if the detector returns an unexpected error.
    """
    base_url = os.environ.get("DETECTOR_URL", _DEFAULT_DETECTOR_URL).rstrip("/")
    delete_url = f"{base_url}/detections/{detection_id}"
    try:
        resp = httpx.delete(delete_url, timeout=_DELETE_TIMEOUT_SEC)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Detector unavailable: {exc}"
        ) from exc
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Detection not found")
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Detector failed to delete detection ({resp.status_code})",
        )
    return Response(status_code=204)
