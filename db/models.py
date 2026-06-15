"""SQLModel ORM models for the BirdFinder database."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class DetectionRecord(SQLModel, table=True):
    """Persisted record for a single high-confidence bird detection.

    Columns:
        id: Auto-incrementing primary key.
        timestamp: Wall-clock time of the detection.
        species: Classified species name.
        confidence: Classification confidence in [0, 1].
        image_path: Path to the full saved image, relative to IMAGE_DIR.
        thumbnail_path: Path to the 200x200 JPEG thumbnail, relative to IMAGE_DIR.
        track_id: Identifier from the stable-detection tracker (nullable for legacy writes).
        stable_frames: Number of consecutive frames the track was stable before classification.
        duration_sec: Approximate track lifetime in seconds (nullable if unavailable).
        uploaded_at: Timestamp of cloud upload; NULL means not yet uploaded.
    """

    __tablename__ = "detections"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime
    species: str
    confidence: float
    image_path: str
    thumbnail_path: str
    track_id: Optional[int] = Field(default=None, nullable=True)
    stable_frames: Optional[int] = Field(default=None, nullable=True)
    duration_sec: Optional[float] = Field(default=None, nullable=True)
    uploaded_at: Optional[datetime] = Field(default=None, nullable=True)
