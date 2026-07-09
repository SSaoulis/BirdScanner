"""Species summary + classifier-vocabulary endpoints for filter/correction UIs."""

import os
from datetime import datetime
from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, col, func, select

from birdscanner.db.models import DetectionRecord, GeoPrior, GeoPriorMeta
from birdscanner.api.dependencies import get_session

router = APIRouter(prefix="/api/species", tags=["species"])

# The classifier's class map lives with the detector (the API image ships no
# assets), so the full species vocabulary is proxied from the detector's control
# server, like the camera/settings routes.
_DEFAULT_DETECTOR_URL = "http://detector:8000"
_DETECTOR_TIMEOUT_SEC = 10.0

# The single fixed metadata row recording the location the priors were built for.
_GEO_META_ID = 1

# Default / maximum number of expected species returned for the current week.
# The default is deliberately small so the Dashboard panel stays a compact,
# glanceable band rather than a long list.
_EXPECTED_DEFAULT_LIMIT = 6
_EXPECTED_MAX_LIMIT = 24


def _current_geo_week(when: datetime) -> int:
    """Map a date to a geomodel week index in ``1..48``.

    Splits each month into quarters (days 1-7 -> 1, 8-14 -> 2, 15-21 -> 3,
    22+ -> 4) and offsets by the month. This is a copy of the canonical
    :func:`birdscanner.ml.geomodel.week_of_year`; it is duplicated here rather
    than imported because ``ml.geomodel`` imports ``onnxruntime`` at module
    scope and the API image ships no ONNX runtime.

    Args:
        when: The date/datetime to bin.

    Returns:
        A week index in ``[1, 48]``.
    """
    quarter = min(4, (when.day - 1) // 7 + 1)
    return (when.month - 1) * 4 + quarter


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


class ExpectedSpecies(BaseModel):
    """One species the geomodel expects at the configured location this week.

    Attributes:
        species: Classifier species label.
        score: Raw geomodel occurrence probability in [0, 1] for the current
            week — an occurrence likelihood, not a "chance you'll see it today".
    """

    species: str
    score: float


class ExpectedThisWeek(BaseModel):
    """The species most likely to be around the feeder for the current week.

    Attributes:
        week: The geomodel week index (1..48) the ranking is for.
        latitude: Latitude the priors were built for, or ``None`` when no
            location is configured (so the client can prompt the user to set one).
        longitude: Longitude the priors were built for, or ``None`` when unset.
        species: Expected species ordered by occurrence score descending; empty
            when no priors are stored.
    """

    week: int
    latitude: Optional[float]
    longitude: Optional[float]
    species: List[ExpectedSpecies]


@router.get("/expected", response_model=ExpectedThisWeek)
def get_expected_this_week(
    limit: int = Query(_EXPECTED_DEFAULT_LIMIT, ge=1, le=_EXPECTED_MAX_LIMIT),
    session: Session = Depends(get_session),
) -> ExpectedThisWeek:
    """Return the species the geomodel expects near the feeder this week.

    Reads the stored spatio-temporal prior (``geo_priors``) for the current
    geomodel week and returns the ``limit`` highest-scoring species. When no
    location is configured (or the prior has never been built) the location
    fields are ``None`` and ``species`` is empty — a clean 200 the UI turns into
    a "set your location" prompt rather than an error.

    Args:
        limit: Maximum number of species to return (1..24, default 12).
        session: Injected read-only database session.

    Returns:
        An ``ExpectedThisWeek`` for the current week.
    """
    week = _current_geo_week(datetime.now())
    rows = session.exec(
        select(GeoPrior)
        .where(GeoPrior.week == week)
        .order_by(col(GeoPrior.probability).desc())
        .limit(limit)
    ).all()
    meta = session.get(GeoPriorMeta, _GEO_META_ID)
    return ExpectedThisWeek(
        week=week,
        latitude=meta.latitude if meta else None,
        longitude=meta.longitude if meta else None,
        species=[
            ExpectedSpecies(species=row.species, score=row.probability) for row in rows
        ],
    )
