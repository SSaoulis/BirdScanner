"""Synchronous deletion of detection records and their image files.

The detector process owns the database and image directory read-write (the API
mounts both read-only), so all deletions are performed here, inside the detector,
and the API proxies delete requests through to it (see
``birdscanner/detector/camera_server.py`` and ``birdscanner/api/routers/detections.py``).

Unlike :class:`db.writer.DetectionWriter`, deletion is performed synchronously:
it runs on the detector's HTTP server thread, not the camera callback, so there
is no need to offload it to a background queue.
"""

import logging
from pathlib import Path

from birdscanner.db.database import SessionFactory
from birdscanner.db.models import DetectionRecord

logger = logging.getLogger("tracking")


def delete_detection(
    session_factory: SessionFactory,
    image_dir: Path,
    detection_id: int,
) -> bool:
    """Delete a detection's database row and its image + thumbnail + video files.

    The image, thumbnail, and video files are removed on a best-effort basis: a
    missing or unremovable file is logged but does not prevent the database row
    from being deleted, so a half-cleaned detection can always be fully removed.

    Args:
        session_factory: Zero-argument callable returning a ``Session`` context
            manager (see :func:`db.database.make_session_factory`).
        image_dir: Root directory under which image paths are stored; the
            record's ``image_path`` / ``thumbnail_path`` are relative to it.
        detection_id: Primary key of the detection to delete.

    Returns:
        ``True`` if a record was found and deleted, ``False`` if no record with
        that id exists.
    """
    with session_factory() as session:
        record = session.get(DetectionRecord, detection_id)
        if record is None:
            return False

        for relative in (record.image_path, record.thumbnail_path, record.video_path):
            if relative:
                _unlink_best_effort(image_dir / relative)

        session.delete(record)
        session.commit()

    logger.info("Deleted detection %d", detection_id)
    return True


def _unlink_best_effort(path: Path) -> None:
    """Delete a file, logging (but not raising) if it cannot be removed.

    Args:
        path: Filesystem path to remove.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete image file %s: %s", path, exc)
