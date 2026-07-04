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
        confidence: Species-classification confidence (ConvNeXt) in [0, 1].
        detection_confidence: Object-detection confidence (YOLO11n on the IMX500)
            for the bird bounding box, in [0, 1] (nullable for legacy rows written
            before the object-detection score was persisted).
        image_path: Path to the full saved image, relative to IMAGE_DIR.
        thumbnail_path: Path to the 200x200 JPEG thumbnail, relative to IMAGE_DIR.
        video_path: Path to the saved short mp4 clip, relative to IMAGE_DIR
            (nullable for legacy rows and rows written before the clip finishes
            encoding — the file appears a few seconds after the row).
        track_id: Identifier from the stable-detection tracker (nullable for legacy writes).
        stable_frames: Number of consecutive frames the track was stable before classification.
        duration_sec: Approximate track lifetime in seconds (nullable if unavailable).
        uploaded_at: Timestamp of cloud upload; NULL means not yet uploaded.
        box_x: Detection box left edge as a fraction [0, 1] of the saved image width
            (nullable for legacy rows written before boxes were persisted).
        box_y: Detection box top edge as a fraction [0, 1] of the saved image height.
        box_w: Detection box width as a fraction [0, 1] of the saved image width.
        box_h: Detection box height as a fraction [0, 1] of the saved image height.
    """

    __tablename__ = "detections"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime
    species: str
    confidence: float
    detection_confidence: Optional[float] = Field(default=None, nullable=True)
    image_path: str
    thumbnail_path: str
    video_path: Optional[str] = Field(default=None, nullable=True)
    track_id: Optional[int] = Field(default=None, nullable=True)
    stable_frames: Optional[int] = Field(default=None, nullable=True)
    duration_sec: Optional[float] = Field(default=None, nullable=True)
    uploaded_at: Optional[datetime] = Field(default=None, nullable=True)
    box_x: Optional[float] = Field(default=None, nullable=True)
    box_y: Optional[float] = Field(default=None, nullable=True)
    box_w: Optional[float] = Field(default=None, nullable=True)
    box_h: Optional[float] = Field(default=None, nullable=True)
