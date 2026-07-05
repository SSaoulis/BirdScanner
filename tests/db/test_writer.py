"""Unit tests for ``DetectionWriter`` (fire-and-forget background DB writes).

The in-memory ``engine`` / ``session_factory`` fixtures come from the top-level
``conftest.py``. Schema-creation and migration behaviour lives in ``test_database.py``.
"""

import time
from datetime import datetime
from typing import List

import pytest
from sqlmodel import Session, select

from birdscanner.db.models import DetectionRecord
from birdscanner.db.writer import DetectionWriter


@pytest.fixture()
def writer(session_factory):
    """DetectionWriter backed by the in-memory engine; stopped after each test."""
    w = DetectionWriter(session_factory)
    yield w
    w.stop()


def _all_records(engine) -> List[DetectionRecord]:
    """Helper: read all rows from the detections table."""
    with Session(engine) as session:
        return list(session.exec(select(DetectionRecord)).all())


def test_write_persists_a_row(writer, engine):
    """A single write() call should result in exactly one row in the DB."""
    ts = datetime(2026, 6, 15, 12, 0, 0)
    writer.write(
        DetectionRecord(
            timestamp=ts,
            species="Parus major",
            confidence=0.92,
            detection_confidence=0.81,
            image_path="Parus major/2026-06-15 12-00-00.png",
            thumbnail_path="Parus major/2026-06-15 12-00-00_thumb.jpg",
            track_id=7,
            stable_frames=5,
        )
    )
    writer.stop()

    records = _all_records(engine)
    assert len(records) == 1
    r = records[0]
    assert r.species == "Parus major"
    assert r.confidence == pytest.approx(0.92)
    assert r.detection_confidence == pytest.approx(0.81)
    assert r.image_path == "Parus major/2026-06-15 12-00-00.png"
    assert r.thumbnail_path == "Parus major/2026-06-15 12-00-00_thumb.jpg"
    assert r.track_id == 7
    assert r.stable_frames == 5
    assert r.uploaded_at is None


def test_write_multiple_rows(writer, engine):
    """Multiple write() calls each produce a separate row."""
    for i in range(3):
        writer.write(
            DetectionRecord(
                timestamp=datetime(2026, 6, 15, 12, i, 0),
                species=f"Species{i}",
                confidence=0.8 + i * 0.05,
                image_path=f"Species{i}/ts.png",
                thumbnail_path=f"Species{i}/ts_thumb.jpg",
            )
        )
    writer.stop()

    records = _all_records(engine)
    assert len(records) == 3
    species_names = {r.species for r in records}
    assert species_names == {"Species0", "Species1", "Species2"}


def test_optional_fields_default_to_none(writer, engine):
    """detection_confidence, track_id, stable_frames, duration_sec, uploaded_at nullable."""
    writer.write(
        DetectionRecord(
            timestamp=datetime.now(),
            species="Turdus merula",
            confidence=0.77,
            image_path="Turdus merula/img.png",
            thumbnail_path="Turdus merula/img_thumb.jpg",
        )
    )
    writer.stop()

    records = _all_records(engine)
    assert len(records) == 1
    r = records[0]
    assert r.detection_confidence is None
    assert r.video_path is None
    assert r.track_id is None
    assert r.stable_frames is None
    assert r.duration_sec is None
    assert r.uploaded_at is None
    assert r.box_x is None
    assert r.box_y is None
    assert r.box_w is None
    assert r.box_h is None


def test_write_persists_normalized_box(writer, engine):
    """The normalized detection box is persisted when supplied."""
    writer.write(
        DetectionRecord(
            timestamp=datetime.now(),
            species="Erithacus rubecula",
            confidence=0.88,
            image_path="Erithacus rubecula/img.png",
            thumbnail_path="Erithacus rubecula/img_thumb.jpg",
            box_x=0.1,
            box_y=0.2,
            box_w=0.3,
            box_h=0.4,
        )
    )
    writer.stop()

    records = _all_records(engine)
    assert len(records) == 1
    r = records[0]
    assert r.box_x == pytest.approx(0.1)
    assert r.box_y == pytest.approx(0.2)
    assert r.box_w == pytest.approx(0.3)
    assert r.box_h == pytest.approx(0.4)


def test_write_persists_video_path(writer, engine):
    """The video clip path is persisted when supplied."""
    writer.write(
        DetectionRecord(
            timestamp=datetime.now(),
            species="Cyanistes caeruleus",
            confidence=0.9,
            image_path="Cyanistes caeruleus/img.png",
            thumbnail_path="Cyanistes caeruleus/img_thumb.jpg",
            video_path="Cyanistes caeruleus/img.mp4",
        )
    )
    writer.stop()

    records = _all_records(engine)
    assert len(records) == 1
    assert records[0].video_path == "Cyanistes caeruleus/img.mp4"


def test_id_autoincrement(writer, engine):
    """Each row gets a unique auto-incremented id."""
    for _ in range(3):
        writer.write(
            DetectionRecord(
                timestamp=datetime.now(),
                species="Parus major",
                confidence=0.9,
                image_path="x/img.png",
                thumbnail_path="x/img_thumb.jpg",
            )
        )
    writer.stop()

    records = _all_records(engine)
    ids = [r.id for r in records]
    assert len(set(ids)) == 3
    assert all(i is not None for i in ids)


def test_write_is_non_blocking(session_factory, engine):
    """write() must return before the DB commit completes (fire-and-forget)."""
    writer = DetectionWriter(session_factory)
    t0 = time.monotonic()
    writer.write(
        DetectionRecord(
            timestamp=datetime.now(),
            species="Pica pica",
            confidence=0.85,
            image_path="Pica pica/img.png",
            thumbnail_path="Pica pica/img_thumb.jpg",
        )
    )
    elapsed = time.monotonic() - t0
    writer.stop()

    # write() should return essentially instantly (well under 100 ms).
    assert elapsed < 0.1
    assert len(_all_records(engine)) == 1


def test_stop_flushes_pending_writes(session_factory, engine):
    """stop() waits for all enqueued records to be committed before returning."""
    writer = DetectionWriter(session_factory)
    count = 10
    for i in range(count):
        writer.write(
            DetectionRecord(
                timestamp=datetime.now(),
                species=f"Bird{i}",
                confidence=0.9,
                image_path=f"Bird{i}/img.png",
                thumbnail_path=f"Bird{i}/img_thumb.jpg",
            )
        )
    writer.stop()

    assert len(_all_records(engine)) == count
