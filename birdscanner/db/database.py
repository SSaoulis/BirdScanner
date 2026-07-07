"""Engine creation and session factory for the BirdFinder SQLite database.

The database path is read from the ``DB_PATH`` environment variable; it
defaults to ``detections.db`` relative to the current working directory.

Typical usage::

    from birdscanner.db.database import make_engine, init_db, make_session_factory
    from birdscanner.db.writer import DetectionWriter

    engine = make_engine()
    init_db(engine)
    writer = DetectionWriter(make_session_factory(engine))
"""

import os
from typing import Callable

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

# Imported for its side effect: importing the model registers it on SQLModel's
# metadata so ``init_db`` can create the table. The block disable survives black
# re-wrapping the import (a line-level disable would drift off the anchor line).
# pylint: disable=unused-import
from birdscanner.db.models import DetectionRecord  # noqa: F401

# pylint: enable=unused-import

_DEFAULT_DB_PATH = "detections.db"

# Type alias: a zero-argument callable that returns an open Session context manager.
SessionFactory = Callable[[], Session]


def make_engine(db_path: str | None = None, *, read_only: bool = False):
    """Create a SQLAlchemy engine for the SQLite database.

    Args:
        db_path: Explicit filesystem path for the SQLite file.  When
            omitted the value of the ``DB_PATH`` environment variable is
            used, falling back to ``detections.db`` in the cwd.
        read_only: When ``True`` the SQLite file is opened in read-only
            mode (``mode=ro``).  This is required when the database lives
            on a read-only mount: a default read-write connection would
            try to create a journal file in the directory and fail with
            ``unable to open database file``.  Read-only mode also requires
            that the file already exist (the writer process must create it).

    Returns:
        A SQLAlchemy ``Engine`` instance.
    """
    path = db_path or os.environ.get("DB_PATH", _DEFAULT_DB_PATH)
    # Read-only mode uses SQLite's URI filename syntax so it never attempts to
    # create the file or a journal alongside it (essential on a read-only mount).
    url = (
        f"sqlite:///file:{path}?mode=ro&uri=true" if read_only else f"sqlite:///{path}"
    )
    # check_same_thread=False is required for SQLite when the session is used
    # from a different thread than the one that created the connection.
    return create_engine(
        url,
        echo=False,
        connect_args={"check_same_thread": False},
    )


# Columns added after the initial schema shipped. ``create_all`` only creates
# missing *tables*, never new columns on an existing one, so a database written
# by an older build keeps its old shape. Each entry is a nullable column that is
# back-filled with ``ALTER TABLE ... ADD COLUMN`` on startup; SQLite adds it with
# a NULL default, which legacy rows carry harmlessly.
_DETECTIONS_ADDED_COLUMNS: dict[str, str] = {
    "detection_confidence": "FLOAT",
    "video_path": "TEXT",
    "no_video_reason": "TEXT",
    "box_x": "FLOAT",
    "box_y": "FLOAT",
    "box_w": "FLOAT",
    "box_h": "FLOAT",
    "classifier_species": "TEXT",
    "classifier_confidence": "FLOAT",
    "geo_scores": "TEXT",
    "corrected": "BOOLEAN",
    "original_species": "TEXT",
}


def _migrate_detections_columns(engine) -> None:
    """Add any newly-introduced nullable columns to an existing detections table.

    SQLModel's ``create_all`` does not alter a table that already exists, so a
    database created by an earlier build is missing columns added since. This
    backfills them in place (no data migration needed — they are nullable).

    Args:
        engine: The SQLAlchemy engine to migrate.
    """
    inspector = inspect(engine)
    if "detections" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("detections")}
    missing = {
        name: sql_type
        for name, sql_type in _DETECTIONS_ADDED_COLUMNS.items()
        if name not in existing
    }
    if not missing:
        return
    with engine.begin() as conn:
        for name, sql_type in missing.items():
            conn.execute(text(f"ALTER TABLE detections ADD COLUMN {name} {sql_type}"))


def init_db(engine) -> None:
    """Create all tables defined by SQLModel metadata if they do not exist.

    Also runs lightweight in-place column migrations for tables that predate
    columns added in later builds (see ``_migrate_detections_columns``).

    Args:
        engine: The SQLAlchemy engine to initialise.
    """
    SQLModel.metadata.create_all(engine)
    _migrate_detections_columns(engine)


def make_session_factory(engine) -> SessionFactory:
    """Return a callable that produces a new ``Session`` context manager.

    Args:
        engine: The SQLAlchemy engine to bind sessions to.

    Returns:
        A zero-argument callable; each call returns a fresh ``Session``.
    """
    return lambda: Session(engine)
