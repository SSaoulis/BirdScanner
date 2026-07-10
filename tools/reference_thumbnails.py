#!/usr/bin/env python3
"""Local reference-thumbnail generation for the species-reference builder.

Split out of ``tools/build_species_reference.py`` (which orchestrates the
network fetch + manifest assembly): these helpers do the pure, offline image
work — center-crop + resize an on-disk reference original into a small square
JPEG served to the gallery/dashboard panels so they never pull a multi-MB
original into a 40px box. No network; only ``os`` + Pillow.

This is a dev-only build tool, **not** part of the BirdScanner runtime.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from PIL import Image  # dev-only tool; Pillow ships in the project .venv

_LOGGER = logging.getLogger(__name__)

# Reference thumbnails: 128px is enough for a crisp retina render at ~40px
# display, at a fraction of the original's bytes.
THUMBNAIL_SIZE = 128
THUMBNAIL_QUALITY = 80


def _thumbnail_rel_path(image_rel_path: str) -> str:
    """Return the ``*_thumb.jpg`` sibling path for an image's relative path.

    Mirrors the detection ``_thumb.jpg`` naming convention: the thumbnail sits
    next to the original with a ``_thumb`` suffix before the extension, e.g.
    ``images/robin/0.jpg`` -> ``images/robin/0_thumb.jpg``.

    Args:
        image_rel_path: The original image path, relative to the bank root.

    Returns:
        The thumbnail path relative to the bank root.
    """
    base, ext = os.path.splitext(image_rel_path)
    return f"{base}_thumb{ext or '.jpg'}"


def make_thumbnail(src_path: str, dest_path: str, size: int = THUMBNAIL_SIZE) -> bool:
    """Write a small center-cropped square JPEG thumbnail of an image.

    Opens the source with Pillow, center-crops to a square, resizes to
    ``size``x``size`` and saves a JPEG. Any failure (unreadable/corrupt source)
    is logged and reported as ``False`` so a single bad image never aborts a
    build — but the warning means a wholesale "Generated 0 thumbnails" can no
    longer silently hide a real, systemic failure.

    Args:
        src_path: Absolute path to the source image on disk.
        dest_path: Absolute destination path for the thumbnail.
        size: Output edge length in pixels.

    Returns:
        ``True`` when the thumbnail was written, ``False`` on any error.
    """
    try:
        with Image.open(src_path) as img:
            square = img.convert("RGB")
            width, height = square.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            square = square.crop((left, top, left + side, top + side))
            square = square.resize((size, size), Image.Resampling.LANCZOS)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            square.save(dest_path, "JPEG", quality=THUMBNAIL_QUALITY)
        return True
    except Exception as exc:
        _LOGGER.warning(
            "Failed to make thumbnail %s -> %s: %s", src_path, dest_path, exc
        )
        return False


def ensure_thumbnails(species_map: dict[str, Any], output_dir: str) -> int:
    """Backfill missing reference thumbnails from the on-disk originals.

    For every manifest image that has an original file on disk but no usable
    thumbnail (``thumbnail_path`` absent, or the thumb file missing), generate
    the ``*_thumb.jpg`` locally and stamp ``thumbnail_path`` on the entry. This
    is a pure local resize (no network), so an already-built bank gains
    thumbnails on a plain re-run without re-downloading anything. Mutates the
    entries in ``species_map`` in place.

    Args:
        species_map: The manifest ``species`` map.
        output_dir: Root of the reference bank; image paths are relative to it.

    Returns:
        The number of thumbnails generated this pass.
    """
    generated = 0
    for entry in species_map.values():
        if not isinstance(entry, dict):
            continue
        for image in entry.get("images", []):
            if not isinstance(image, dict):
                continue
            rel_path = image.get("path")
            if not isinstance(rel_path, str) or not rel_path:
                continue
            if not os.path.exists(os.path.join(output_dir, rel_path)):
                continue
            existing_thumb = image.get("thumbnail_path")
            if (
                isinstance(existing_thumb, str)
                and existing_thumb
                and os.path.exists(os.path.join(output_dir, existing_thumb))
            ):
                continue
            thumb_rel = _thumbnail_rel_path(rel_path)
            if make_thumbnail(
                os.path.join(output_dir, rel_path),
                os.path.join(output_dir, thumb_rel),
            ):
                image["thumbnail_path"] = thumb_rel
                generated += 1
    return generated
