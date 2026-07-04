"""Shared FastAPI dependency providers for database sessions and config."""

import os
from pathlib import Path
from typing import Generator

from sqlmodel import Session

from birdscanner.db.database import make_engine, make_session_factory

_DEFAULT_IMAGE_DIR = "/home/stefan/Pictures/bird_detections"

# Repo-relative default for the offline-built species reference data bank.
# Resolved the same robust way ``birdscanner/api/main.py`` resolves the frontend dist
# directory so it works regardless of the process's current working directory.
_DEFAULT_REFERENCE_DIR = Path(__file__).parent.parent / "assets" / "species_reference"

# Module-level singletons so the engine is created once per process.  These are
# mutable, lazily-initialised caches (not true constants), so the lower-case
# leading-underscore names are intentional.
_engine = None  # pylint: disable=invalid-name
_session_factory = None  # pylint: disable=invalid-name


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


def get_reference_dir() -> Path:
    """FastAPI dependency that returns the species-reference root directory.

    Reads ``SPECIES_REFERENCE_DIR`` from the environment; defaults to the
    repo-relative ``assets/species_reference`` directory.  The directory holds
    the offline-built ``manifest.json`` plus cached reference images and may not
    exist yet (the reference API degrades gracefully when it is absent).

    Returns:
        Resolved ``Path`` to the species-reference directory.
    """
    env_value = os.environ.get("SPECIES_REFERENCE_DIR")
    return Path(env_value) if env_value else _DEFAULT_REFERENCE_DIR
