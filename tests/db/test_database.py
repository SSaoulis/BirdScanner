"""Tests for engine creation, schema init, and in-place column migration.

Uses temporary on-disk / in-memory SQLite databases so no real data volume is
needed. The shared in-memory ``engine`` fixture (from the top-level conftest) has
already run ``init_db``.
"""

from datetime import datetime

import pytest
from sqlalchemy import inspect
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select, text

from birdscanner.db.database import (
    init_db,
    make_engine,
    make_session_factory,
)
from birdscanner.db.models import DetectionRecord

_EXPECTED_COLUMNS = {
    "id",
    "timestamp",
    "species",
    "confidence",
    "detection_confidence",
    "image_path",
    "thumbnail_path",
    "video_path",
    "no_video_reason",
    "track_id",
    "stable_frames",
    "duration_sec",
    "uploaded_at",
    "box_x",
    "box_y",
    "box_w",
    "box_h",
    "classifier_species",
    "classifier_confidence",
    "geo_scores",
    "corrected",
    "original_species",
}


def test_init_db_creates_table_with_expected_columns(engine):
    """init_db creates the detections table with the full current column set."""
    with Session(engine) as session:
        result = session.exec(  # type: ignore[call-overload]
            text("PRAGMA table_info(detections)")
        ).all()
    col_names = {row[1] for row in result}
    assert col_names == _EXPECTED_COLUMNS


def test_init_db_backfills_added_columns_on_legacy_table():
    """init_db backfills detection_confidence + video_path + box_* + correction columns."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    # Create a legacy detections table without the columns added post-launch.
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE detections ("
                "id INTEGER PRIMARY KEY, timestamp DATETIME, species TEXT, "
                "confidence FLOAT, image_path TEXT, thumbnail_path TEXT)"
            )
        )

    init_db(engine)

    col_names = {col["name"] for col in inspect(engine).get_columns("detections")}
    assert {
        "detection_confidence",
        "video_path",
        "no_video_reason",
        "box_x",
        "box_y",
        "box_w",
        "box_h",
        "corrected",
        "original_species",
    } <= col_names


def test_init_db_is_idempotent(engine):
    """Running init_db again on an already-migrated DB is a harmless no-op."""
    init_db(engine)  # second call
    col_names = {col["name"] for col in inspect(engine).get_columns("detections")}
    assert col_names == _EXPECTED_COLUMNS


def test_make_session_factory_returns_fresh_sessions(engine):
    """Each call to the factory yields a distinct, usable Session."""
    factory = make_session_factory(engine)
    with factory() as s1, factory() as s2:
        assert isinstance(s1, Session)
        assert s1 is not s2


def test_make_engine_read_only_opens_existing_file(tmp_path):
    """A read-only engine can query a file it did not create, without a journal."""
    db_path = tmp_path / "detections.db"

    # Writer process creates + populates the DB.
    rw_engine = make_engine(str(db_path))
    init_db(rw_engine)
    with Session(rw_engine) as session:
        session.add(
            DetectionRecord(
                timestamp=datetime(2024, 6, 1, 12, 0, 0),
                species="Robin",
                confidence=0.9,
                image_path="Robin/a.png",
                thumbnail_path="Robin/a_thumb.jpg",
            )
        )
        session.commit()
    rw_engine.dispose()

    # Read-only consumer opens the same file with mode=ro.
    ro_engine = make_engine(str(db_path), read_only=True)
    with Session(ro_engine) as session:
        rows = list(session.exec(select(DetectionRecord)).all())
    assert len(rows) == 1
    assert rows[0].species == "Robin"


def test_make_engine_read_only_missing_file_errors_on_use(tmp_path):
    """Read-only mode never creates the file, so querying a missing DB fails."""
    ro_engine = make_engine(str(tmp_path / "does_not_exist.db"), read_only=True)
    with pytest.raises(Exception):
        with Session(ro_engine) as session:
            session.exec(select(DetectionRecord)).all()
    # The file must not have been created by the read-only connection.
    assert not (tmp_path / "does_not_exist.db").exists()
