"""Persistence for user-added ("custom") species labels.

The classifier's vocabulary is fixed by the model, but a bird it was never trained
on can still be seen at the feeder. When the user re-identifies such a detection to
a species the model does not know (from the Lightbox correction picker), the new
label is recorded here so it persists across restarts and reappears in the picker
for reuse. The detector then serves the union of the classifier's classes and these
rows as the species vocabulary (see the control server's ``GET /labels``).

The detector process owns the database read-write, so — like the deleter and the
corrector (see :mod:`birdscanner.db.deleter` / :mod:`birdscanner.db.corrector`) —
these writes happen here, synchronously, on the control server's HTTP thread (the
API mounts the DB read-only). This module only reads and writes the database; it has
no ``ml``/camera dependencies.
"""

import logging
from datetime import datetime

from sqlmodel import select

from birdscanner.db.database import SessionFactory
from birdscanner.db.models import CustomSpecies

logger = logging.getLogger("tracking")


def list_custom_species(session_factory: SessionFactory) -> list[str]:
    """Return every stored custom species label.

    Args:
        session_factory: Zero-argument callable returning a ``Session`` context
            manager (see :func:`db.database.make_session_factory`).

    Returns:
        The custom species labels (unsorted); an empty list when none are stored.
    """
    with session_factory() as session:
        return list(session.exec(select(CustomSpecies.name)).all())


def add_custom_species(session_factory: SessionFactory, name: str) -> str:
    """Add a custom species label, de-duplicating case-insensitively.

    The label is stored trimmed and exact-cased. If a label already exists that
    matches ``name`` case-insensitively, no new row is written and that existing
    (canonical) label is returned unchanged — so ``"hoopoe"`` never creates a second
    entry alongside ``"Hoopoe"``. This makes the call idempotent.

    Args:
        session_factory: Zero-argument callable returning a ``Session`` context
            manager.
        name: The species label to add. Must be non-empty after trimming (callers
            validate and bound its length upstream — e.g. the control handler).

    Returns:
        The canonical stored label (the pre-existing one on a case-insensitive
        match, otherwise the newly stored, trimmed ``name``).

    Raises:
        ValueError: If ``name`` is empty or whitespace-only.
    """
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("Custom species name must not be empty")

    lowered = trimmed.lower()
    with session_factory() as session:
        for existing in session.exec(select(CustomSpecies.name)).all():
            if existing.lower() == lowered:
                return existing

        session.add(CustomSpecies(name=trimmed, created_at=datetime.now()))
        session.commit()

    logger.info("Added custom species label: %s", trimmed)
    return trimmed
