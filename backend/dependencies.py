"""Shared FastAPI dependency providers for database sessions and config."""

import os
from pathlib import Path
from typing import Generator

from sqlmodel import Session

from db.database import make_engine, make_session_factory

_DEFAULT_IMAGE_DIR = "/home/stefan/Pictures/bird_detections"

# Module-level singletons so the engine is created once per process.
_engine = None
_session_factory = None


def _get_engine():
    """Return the module-level SQLAlchemy engine, creating it on first call.

    The API is a read-only consumer: it mounts the database read-only and the
    detector service owns all writes (including schema creation).  The engine is
    therefore opened in read-only mode and the API never runs ``init_db``.
    """
    global _engine
    if _engine is None:
        _engine = make_engine(read_only=True)
    return _engine


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request.

    Yields:
        An open ``sqlmodel.Session`` that is closed after the response is sent.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = make_session_factory(_get_engine())
    with _session_factory() as session:
        yield session


def get_image_dir() -> Path:
    """FastAPI dependency that returns the image root directory as a ``Path``.

    Reads ``IMAGE_DIR`` from the environment; defaults to
    ``/home/stefan/Pictures/bird_detections``.

    Returns:
        Resolved ``Path`` to the image directory.
    """
    return Path(os.environ.get("IMAGE_DIR", _DEFAULT_IMAGE_DIR))
