"""Synchronous correction of a detection's classified species.

A user who disagrees with the classifier can reassign a detection to a different
species from the Lightbox panel.  Like deletion (see ``birdscanner/db/deleter.py``),
the write happens here inside the detector — it owns the database and image
directory read-write, while the API mounts both read-only and proxies the request
through the detector's control server (``birdscanner/detector/camera_server.py`` and
``birdscanner/api/routers/detections.py``).

Correcting a detection:

* moves the saved still / thumbnail / clip into the corrected species' folder
  (best-effort — a missing or unmovable file does not block the row update, so a
  half-organised detection can always be re-labelled),
* rewrites the stored relative paths to the new species folder,
* records the model's original guess in ``original_species`` (kept for retraining
  ground truth) the first time a detection is corrected, and
* flags the row ``corrected`` while keeping the model's original ``confidence``.

The species value is trusted here: the caller (the control-server handler) has
already validated it against the classifier vocabulary, so this module does not
re-check it.
"""

import logging
from pathlib import Path
from typing import Optional

from birdscanner.db.database import SessionFactory
from birdscanner.db.models import DetectionRecord

logger = logging.getLogger("tracking")

# The relative-path fields that embed the species folder as their first segment.
_IMAGE_PATH_FIELDS = ("image_path", "thumbnail_path", "video_path")


def correct_detection_species(
    session_factory: SessionFactory,
    image_dir: Path,
    detection_id: int,
    new_species: str,
) -> Optional[dict]:
    """Reassign a detection to ``new_species`` and move its files accordingly.

    Args:
        session_factory: Zero-argument callable returning a ``Session`` context
            manager (see :func:`db.database.make_session_factory`).
        image_dir: Root directory under which image paths are stored; the
            record's ``image_path`` / ``thumbnail_path`` / ``video_path`` are
            relative to it.
        detection_id: Primary key of the detection to correct.
        new_species: The user-chosen species label to assign (already validated
            against the classifier vocabulary by the caller).

    Returns:
        The updated record as a JSON-serialisable ``dict`` (via
        ``model_dump(mode="json")``), or ``None`` if no record with that id
        exists.  When ``new_species`` already matches the current species the row
        is returned unchanged (no move, no flag).
    """
    with session_factory() as session:
        record = session.get(DetectionRecord, detection_id)
        if record is None:
            return None

        if new_species == record.species:
            # A no-op correction (same species): nothing to move or flag.
            return record.model_dump(mode="json")

        # Preserve the model's *first* guess across repeat corrections.
        if record.original_species is None:
            record.original_species = record.species

        for field_name in _IMAGE_PATH_FIELDS:
            relative = getattr(record, field_name)
            if not relative:
                continue
            new_relative = f"{new_species}/{Path(relative).name}"
            _move_best_effort(image_dir / relative, image_dir / new_relative)
            setattr(record, field_name, new_relative)

        record.species = new_species
        record.corrected = True

        session.add(record)
        session.commit()
        session.refresh(record)
        result = record.model_dump(mode="json")

    logger.info("Corrected detection %d -> species=%s", detection_id, new_species)
    return result


def _move_best_effort(src: Path, dest: Path) -> None:
    """Move ``src`` to ``dest``, logging (but not raising) on failure.

    A missing source is skipped silently (legacy rows may lack a file); an
    existing destination is overwritten.  Any ``OSError`` (e.g. a cross-filesystem
    move) is logged and swallowed so a file that cannot be moved never blocks the
    species correction — the stored path is updated regardless, and a missing file
    degrades to the gallery's existing 404 fallback.

    Args:
        src: Current filesystem path of the file.
        dest: Target filesystem path under the corrected species' folder.
    """
    try:
        if not src.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        src.replace(dest)
    except OSError as exc:
        logger.warning("Could not move image file %s -> %s: %s", src, dest, exc)
