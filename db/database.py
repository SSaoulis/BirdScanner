"""Engine creation and session factory for the BirdFinder SQLite database.

The database path is read from the ``DB_PATH`` environment variable; it
defaults to ``detections.db`` relative to the current working directory.

Typical usage::

    from db.database import make_engine, init_db, make_session_factory
    from db.writer import DetectionWriter

    engine = make_engine()
    init_db(engine)
    writer = DetectionWriter(make_session_factory(engine))
"""

import os
from typing import Callable

from sqlmodel import Session, SQLModel, create_engine

# Imported for its side effect: importing the model registers it on SQLModel's
# metadata so ``init_db`` can create the table.
from db.models import DetectionRecord  # noqa: F401  # pylint: disable=unused-import

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


def init_db(engine) -> None:
    """Create all tables defined by SQLModel metadata if they do not exist.

    Args:
        engine: The SQLAlchemy engine to initialise.
    """
    SQLModel.metadata.create_all(engine)


def make_session_factory(engine) -> SessionFactory:
    """Return a callable that produces a new ``Session`` context manager.

    Args:
        engine: The SQLAlchemy engine to bind sessions to.

    Returns:
        A zero-argument callable; each call returns a fresh ``Session``.
    """
    return lambda: Session(engine)
