"""Species summary endpoint for filter dropdowns."""

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, func, select

from db.models import DetectionRecord
from backend.dependencies import get_session

router = APIRouter(prefix="/api/species", tags=["species"])


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
    rows = session.exec(
        select(DetectionRecord.species, func.count(DetectionRecord.id).label("count"))
        .group_by(DetectionRecord.species)
        .order_by(func.count(DetectionRecord.id).desc())
    ).all()
    return [SpeciesSummary(species=row[0], count=row[1]) for row in rows]
