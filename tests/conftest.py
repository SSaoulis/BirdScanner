"""Global test fixtures shared across every test package.

Anything here is available to all tests without an explicit import. Layer-specific
fixtures live in the nearest ``conftest.py`` (``tests/api``, ``tests/detector``,
``tests/ml``) so each fixture sits at the lowest scope that covers all its users.

The database / detection-record fixtures live here rather than in ``tests/db``
because both the ``db`` and the ``api`` test packages need them, and sibling
``conftest.py`` files are not visible across packages.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from birdscanner.db.database import init_db, make_session_factory
from birdscanner.db.models import DetectionRecord

FrameValue = Union[int, Sequence[int]]


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


@pytest.fixture()
def frame_factory() -> Callable[..., np.ndarray]:
    """Return a builder for solid-colour RGB frames.

    The returned callable takes ``value`` (a scalar or ``(r, g, b)`` tuple used to
    fill the frame) and ``size`` (``(height, width)``), defaulting to a 16x16 frame
    filled with zeros. A scalar ``value`` doubles as an identity tag for tests that
    need to tell frames apart.
    """

    def _make(value: FrameValue = 0, size: Tuple[int, int] = (16, 16)) -> np.ndarray:
        height, width = size
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = value
        return frame

    return _make


# ---------------------------------------------------------------------------
# Database + detection records
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """In-memory SQLite engine with the schema initialised.

    ``StaticPool`` shares one underlying connection across threads, which is the
    only way an in-memory database persists across the background writer thread
    and the request threads.
    """
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(eng)
    return eng


@pytest.fixture()
def session_factory(engine):
    """Session factory bound to the in-memory test engine."""
    return make_session_factory(engine)


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Temporary directory that stands in for ``IMAGE_DIR``."""
    return tmp_path


@pytest.fixture()
def detection_factory(session_factory, image_dir) -> Callable[..., DetectionRecord]:
    """Return a factory that inserts a detection row + its on-disk stub files.

    Each call writes a fake image + thumbnail (and, when ``with_video`` is set, an
    mp4) under ``image_dir`` and inserts a matching ``DetectionRecord``. ``track_id``
    discriminates the filenames, so callers must pass distinct ids for multiple
    rows in one directory. The refreshed record (with ``id`` populated) is returned.
    """

    def _make(
        *,
        species: str = "Robin",
        confidence: float = 0.95,
        detection_confidence: Optional[float] = 0.8,
        track_id: int = 1,
        ts: Optional[datetime] = None,
        stable_frames: Optional[int] = 5,
        duration_sec: Optional[float] = 1.2,
        with_video: bool = False,
        video_missing: bool = False,
    ) -> DetectionRecord:
        ts = ts or datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        species_dir = image_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)

        img_rel = f"{species}/img_{track_id}.jpg"
        thumb_rel = f"{species}/img_{track_id}_thumb.jpg"
        (image_dir / img_rel).write_bytes(b"FAKEJPEG")
        (image_dir / thumb_rel).write_bytes(b"FAKETHUMB")

        video_rel: Optional[str] = None
        if with_video:
            video_rel = f"{species}/img_{track_id}.mp4"
            if not video_missing:
                (image_dir / video_rel).write_bytes(b"FAKEMP4")

        record = DetectionRecord(
            timestamp=ts,
            species=species,
            confidence=confidence,
            detection_confidence=detection_confidence,
            image_path=img_rel,
            thumbnail_path=thumb_rel,
            video_path=video_rel,
            track_id=track_id,
            stable_frames=stable_frames,
            duration_sec=duration_sec,
        )
        with session_factory() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record

    return _make
