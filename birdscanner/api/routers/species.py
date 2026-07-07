"""Species summary + classifier-vocabulary endpoints for filter/correction UIs."""

import os
from typing import List

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, col, func, select

from birdscanner.db.models import DetectionRecord
from birdscanner.api.dependencies import get_session

router = APIRouter(prefix="/api/species", tags=["species"])

# The classifier's class map lives with the detector (the API image ships no
# assets), so the full species vocabulary is proxied from the detector's control
# server, like the camera/settings routes.
_DEFAULT_DETECTOR_URL = "http://detector:8000"
_DETECTOR_TIMEOUT_SEC = 10.0


class SpeciesSummary(BaseModel):
    """Detection count for a single species.

    Attributes:
        species: Species name as stored in the database.
        count: Total number of detections for this species.
    """

    species: str
    count: int


@router.get("", response_model=List[SpeciesSummary])
def list_species(session: Session = Depends(get_session)) -> List[SpeciesSummary]:
    """Return all observed species with their total detection counts.

    Results are ordered by count descending so the most frequently seen
    species appear first (useful for populating filter dropdowns).

    Args:
        session: Injected database session.

    Returns:
        List of ``SpeciesSummary`` objects sorted by count descending.
    """
    # pylint: disable=not-callable  # sqlalchemy's func.count is a dynamic callable
    rows = session.exec(
        select(
            DetectionRecord.species, func.count(col(DetectionRecord.id)).label("count")
        )
        .group_by(DetectionRecord.species)
        .order_by(func.count(col(DetectionRecord.id)).desc())
    ).all()
    return [SpeciesSummary(species=row[0], count=row[1]) for row in rows]


@router.get("/vocabulary", response_model=List[str])
def get_vocabulary() -> List[str]:
    """Return the classifier's full species vocabulary (for the correction picker).

    Proxies the detector's ``GET /labels`` — the detector owns the classifier
    class map, and the API image ships no assets.  The list is every real species
    the model can predict (the ``Unknown`` sentinel is excluded upstream).

    Returns:
        The sorted list of valid species labels.

    Raises:
        HTTPException: 503 if the detector is unreachable (e.g. the camera is
            still coming up), so the picker can show an offline state.
    """
    base_url = os.environ.get("DETECTOR_URL", _DEFAULT_DETECTOR_URL).rstrip("/")
    try:
        resp = httpx.get(f"{base_url}/labels", timeout=_DETECTOR_TIMEOUT_SEC)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Species list unavailable: {exc}"
        ) from exc
    return resp.json().get("species", [])
