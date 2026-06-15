"""Detection list and detail endpoints."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from db.models import DetectionRecord
from backend.dependencies import get_session

router = APIRouter(prefix="/api/detections", tags=["detections"])


@router.get("", response_model=List[DetectionRecord])
def list_detections(
    species: Optional[str] = Query(default=None, description="Filter by species name"),
    from_: Optional[datetime] = Query(default=None, alias="from", description="Earliest timestamp (inclusive)"),
    to: Optional[datetime] = Query(default=None, description="Latest timestamp (inclusive)"),
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    session: Session = Depends(get_session),
) -> List[DetectionRecord]:
    """Return a paginated, optionally filtered list of detection records.

    Args:
        species: If provided, only return detections for this species.
        from_: If provided, only return detections at or after this timestamp.
        to: If provided, only return detections at or before this timestamp.
        limit: Maximum number of records to return (1–500, default 50).
        offset: Number of records to skip for pagination.
        session: Injected database session.

    Returns:
        List of ``DetectionRecord`` objects ordered by timestamp descending.
    """
    query = select(DetectionRecord)
    if species is not None:
        query = query.where(DetectionRecord.species == species)
    if from_ is not None:
        query = query.where(DetectionRecord.timestamp >= from_)
    if to is not None:
        query = query.where(DetectionRecord.timestamp <= to)
    query = query.order_by(DetectionRecord.timestamp.desc()).offset(offset).limit(limit)  # type: ignore[attr-defined]
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
