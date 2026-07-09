"""Detection list, detail, and delete endpoints."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlmodel import Session, select

from birdscanner.db.models import DetectionRecord
from birdscanner.api.dependencies import get_session
from birdscanner.api.detector_proxy import detector_error_detail

router = APIRouter(prefix="/api/detections", tags=["detections"])

# The API mounts the database and images read-only, so it cannot mutate them
# itself.  Deletes and species corrections are proxied to the detector's control
# server (it owns the data volume read-write), mirroring the camera-snapshot proxy.
_DEFAULT_DETECTOR_URL = "http://detector:8000"
_DETECTOR_TIMEOUT_SEC = 10.0


class SpeciesCorrection(BaseModel):
    """Request body for correcting a detection's classified species.

    Attributes:
        species: The user-chosen species label to assign to the detection.
        allow_new: When ``True``, permit ``species`` to be a brand-new label not in
            the classifier vocabulary — the detector registers it as a custom
            species (see ``birdscanner/db/custom_species.py``). When ``False`` (the
            default), an unknown label is rejected, preserving the correction path's
            typo safety.
    """

    species: str
    allow_new: bool = False


class DetectionFilters:
    """Query-parameter filters for the detections list endpoint.

    Grouping the filters into one dependency keeps the endpoint signature short
    and gives the filter-to-SQL translation a single home (:meth:`apply`).
    """

    def __init__(
        self,
        species: Optional[str] = Query(
            default=None, description="Filter by species name"
        ),
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
    ) -> None:
        """Capture the request's filter query parameters.

        Args:
            species: Optional species-name filter.
            from_: Optional inclusive earliest timestamp (``from`` query alias).
            to: Optional inclusive latest timestamp.
            min_confidence: Optional 0–1 floor on the classification confidence.
        """
        self.species = species
        self.from_ = from_
        self.to = to
        self.min_confidence = min_confidence

    def apply(self, query):
        """Return ``query`` narrowed by whichever filters were supplied.

        Args:
            query: The base ``select(DetectionRecord)`` statement.

        Returns:
            The statement with a ``where`` clause per supplied filter.
        """
        if self.species is not None:
            query = query.where(DetectionRecord.species == self.species)
        if self.from_ is not None:
            query = query.where(DetectionRecord.timestamp >= self.from_)
        if self.to is not None:
            query = query.where(DetectionRecord.timestamp <= self.to)
        if self.min_confidence is not None:
            query = query.where(DetectionRecord.confidence >= self.min_confidence)
        return query


@router.get("", response_model=List[DetectionRecord])
def list_detections(
    filters: DetectionFilters = Depends(),
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    session: Session = Depends(get_session),
) -> List[DetectionRecord]:
    """Return a paginated, optionally filtered list of detection records.

    Args:
        filters: Injected filter query parameters (see :class:`DetectionFilters`).
        limit: Maximum number of records to return (1–500, default 50).
        offset: Number of records to skip for pagination.
        session: Injected database session.

    Returns:
        List of ``DetectionRecord`` objects ordered by timestamp descending,
        with ``id`` descending as a tiebreaker for a deterministic page order.
    """
    query = filters.apply(select(DetectionRecord))
    # Order by timestamp, then id, so rows with identical timestamps keep a
    # stable order across paginated requests. Without the id tiebreaker SQLite
    # may return tied rows in a different order per query, which makes
    # offset-based pages overlap and surfaces duplicate detections in the UI.
    query = (
        query.order_by(
            DetectionRecord.timestamp.desc(),  # type: ignore[attr-defined]  # pylint: disable=no-member
            DetectionRecord.id.desc(),  # type: ignore[union-attr]  # pylint: disable=no-member
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
    owns them (see ``birdscanner/detector/camera_server.py``).  This endpoint forwards the
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
        resp = httpx.delete(delete_url, timeout=_DETECTOR_TIMEOUT_SEC)
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


@router.patch("/{detection_id}")
def correct_detection(
    detection_id: int, correction: SpeciesCorrection
) -> Dict[str, Any]:
    """Correct a detection's species by proxying to the detector's control server.

    The API cannot write the database or move image files (it mounts both
    read-only); the detector owns them (see ``birdscanner/db/corrector.py``).
    This endpoint forwards the correction and relays the detector's updated record
    JSON verbatim (like the settings/crop proxies), so the browser gets the moved
    image paths and the ``corrected`` flag without a re-serialization round-trip.

    Args:
        detection_id: Primary key of the detection to correct.
        correction: The new species to assign.

    Returns:
        The updated detection record as a JSON object (with ``corrected`` set and
        the moved image paths).

    Raises:
        HTTPException: 400 (relaying the detector's message) for an unknown
            species, 404 if the detection does not exist, 503 if the detector is
            unreachable.
    """
    base_url = os.environ.get("DETECTOR_URL", _DEFAULT_DETECTOR_URL).rstrip("/")
    patch_url = f"{base_url}/detections/{detection_id}"
    try:
        resp = httpx.patch(
            patch_url,
            json={"species": correction.species, "allow_new": correction.allow_new},
            timeout=_DETECTOR_TIMEOUT_SEC,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Detector unavailable: {exc}"
        ) from exc
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Detection not found")
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code, detail=detector_error_detail(resp)
        )
    return resp.json()
