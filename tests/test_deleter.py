"""Tests for db.deleter.delete_detection.

Uses an in-memory SQLite database (shared across threads via StaticPool) and a
temporary image directory, so no real detector or filesystem layout is needed.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

from birdscanner.db.database import make_session_factory
from birdscanner.db.deleter import delete_detection
from birdscanner.db.models import DetectionRecord


@pytest.fixture()
def session_factory():
    """In-memory SQLite session factory shared across threads."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    return make_session_factory(engine)


def _seed(session_factory, image_dir: Path) -> int:
    """Insert a detection with on-disk image + thumbnail + video; return its id."""
    species_dir = image_dir / "Robin"
    species_dir.mkdir(parents=True, exist_ok=True)
    img_rel = "Robin/img_1.jpg"
    thumb_rel = "Robin/img_1_thumb.jpg"
    video_rel = "Robin/img_1.mp4"
    (image_dir / img_rel).write_bytes(b"FAKEJPEG")
    (image_dir / thumb_rel).write_bytes(b"FAKETHUMB")
    (image_dir / video_rel).write_bytes(b"FAKEMP4")
    record = DetectionRecord(
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        species="Robin",
        confidence=0.95,
        image_path=img_rel,
        thumbnail_path=thumb_rel,
        video_path=video_rel,
    )
    with session_factory() as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        assert record.id is not None
        return record.id


def test_delete_removes_row_and_files(session_factory, tmp_path: Path):
    det_id = _seed(session_factory, tmp_path)

    assert delete_detection(session_factory, tmp_path, det_id) is True

    with session_factory() as session:
        assert session.get(DetectionRecord, det_id) is None
    assert not (tmp_path / "Robin/img_1.jpg").exists()
    assert not (tmp_path / "Robin/img_1_thumb.jpg").exists()
    assert not (tmp_path / "Robin/img_1.mp4").exists()


def test_delete_missing_record_returns_false(session_factory, tmp_path: Path):
    assert delete_detection(session_factory, tmp_path, 99999) is False


def test_delete_succeeds_when_image_files_already_gone(
    session_factory, tmp_path: Path
):
    det_id = _seed(session_factory, tmp_path)
    # Remove the files out from under the deleter; the row must still be deleted.
    (tmp_path / "Robin/img_1.jpg").unlink()
    (tmp_path / "Robin/img_1_thumb.jpg").unlink()

    assert delete_detection(session_factory, tmp_path, det_id) is True
    with session_factory() as session:
        assert session.get(DetectionRecord, det_id) is None
