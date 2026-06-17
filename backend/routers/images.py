"""Image serving endpoints: thumbnail, full-resolution, and bulk ZIP download."""

import io
import zipfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from db.models import DetectionRecord
from backend.dependencies import get_session, get_image_dir

router = APIRouter(prefix="/api/images", tags=["images"])

_THUMB_MEDIA_TYPE = "image/jpeg"
_JPEG_MEDIA_TYPE = "image/jpeg"


def _resolve_image(
    detection_id: int,
    session: Session,
    image_dir: Path,
    use_thumbnail: bool,
) -> Path:
    """Resolve the filesystem path for a detection's image or thumbnail.

    Args:
        detection_id: Primary key of the detection record.
        session: Database session.
        image_dir: Root directory under which images are stored.
        use_thumbnail: If ``True`` use ``thumbnail_path``; otherwise use ``image_path``.

    Returns:
        Absolute ``Path`` to the image file.

    Raises:
        HTTPException: 404 if the detection does not exist or the file is missing.
    """
    record = session.get(DetectionRecord, detection_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    relative = record.thumbnail_path if use_thumbnail else record.image_path
    path = image_dir / relative
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found on disk")
    return path


@router.get("/{detection_id}/thumbnail")
def get_thumbnail(
    detection_id: int,
    session: Session = Depends(get_session),
    image_dir: Path = Depends(get_image_dir),
) -> StreamingResponse:
    """Serve the 200×200 JPEG thumbnail for a detection.

    Args:
        detection_id: Primary key of the detection.
        session: Injected database session.
        image_dir: Injected image root directory.

    Returns:
        JPEG image response.
    """
    path = _resolve_image(detection_id, session, image_dir, use_thumbnail=True)
    return StreamingResponse(open(path, "rb"), media_type=_THUMB_MEDIA_TYPE)


@router.get("/{detection_id}/full")
def get_full_image(
    detection_id: int,
    session: Session = Depends(get_session),
    image_dir: Path = Depends(get_image_dir),
) -> StreamingResponse:
    """Serve the full-resolution image for a detection.

    Args:
        detection_id: Primary key of the detection.
        session: Injected database session.
        image_dir: Injected image root directory.

    Returns:
        JPEG image response.
    """
    path = _resolve_image(detection_id, session, image_dir, use_thumbnail=False)
    return StreamingResponse(open(path, "rb"), media_type=_JPEG_MEDIA_TYPE)


def _stream_zip(paths: List[Path]):
    """Generator that yields ZIP archive chunks for a list of image files.

    Streams a ZIP in memory without requiring the full archive to be built
    before sending.  Each file is added uncompressed (``ZIP_STORED``) to keep
    CPU overhead negligible on the Pi.

    Args:
        paths: Filesystem paths to include in the archive.

    Yields:
        ``bytes`` chunks of the in-progress ZIP stream.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(
        buf, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as zf:
        for path in paths:
            zf.write(path, arcname=path.name)
            buf.seek(0)
            yield buf.read()
            buf.seek(0)
            buf.truncate()
    buf.seek(0)
    remaining = buf.read()
    if remaining:
        yield remaining


@router.get("/download")
def download_images(
    ids: str = Query(..., description="Comma-separated detection IDs to download"),
    session: Session = Depends(get_session),
    image_dir: Path = Depends(get_image_dir),
) -> StreamingResponse:
    """Stream a ZIP archive of full-resolution images for the given detection IDs.

    Missing detections or files are silently skipped so a partial result is
    always returned rather than failing the whole download.

    Args:
        ids: Comma-separated list of detection primary keys (e.g. ``"1,2,3"``).
        session: Injected database session.
        image_dir: Injected image root directory.

    Returns:
        ``StreamingResponse`` with ``Content-Type: application/zip``.

    Raises:
        HTTPException: 400 if ``ids`` cannot be parsed as integers.
    """
    try:
        id_list: List[int] = [int(i.strip()) for i in ids.split(",") if i.strip()]
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="ids must be comma-separated integers"
        ) from exc

    paths: List[Path] = []
    for detection_id in id_list:
        record = session.get(DetectionRecord, detection_id)
        if record is None:
            continue
        path = image_dir / record.image_path
        if path.is_file():
            paths.append(path)

    headers = {"Content-Disposition": "attachment; filename=bird_detections.zip"}
    return StreamingResponse(
        _stream_zip(paths),
        media_type="application/zip",
        headers=headers,
    )
