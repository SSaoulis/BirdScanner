"""Read-only species-reference API.

Serves the offline-built "species reference" data bank (Wikipedia/Wikidata
summaries plus cached images) from ``assets/species_reference/``.  The manifest
is loaded lazily and memoized per reference directory; a missing or invalid
manifest is treated as "no species have references" so every lookup returns a
clean 404 rather than crashing the API on startup or per request.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from birdscanner.api.dependencies import get_reference_dir

router = APIRouter(prefix="/api/species", tags=["reference"])

_MANIFEST_FILENAME = "manifest.json"

# Memoized manifests keyed by resolved reference directory.  Guarded by a lock
# so concurrent requests do not race on the first load.
_manifest_cache: Dict[str, Optional[Dict[str, Any]]] = {}
_cache_lock = threading.Lock()


class ReferenceImage(BaseModel):
    """A single reference image as exposed to API clients.

    The on-disk ``path`` and upstream ``source_url`` are deliberately not
    exposed; ``url`` instead points at the image-serving endpoint.

    Attributes:
        url: API path that serves the JPEG bytes for this image.
        attribution: Human-readable attribution string for the image.
        license: SPDX-style or human-readable licence string, if known.
    """

    url: str
    attribution: str
    license: Optional[str]


class SpeciesReference(BaseModel):
    """Reference information for a single species.

    Attributes:
        common_name: Species common name (the manifest key).
        scientific_name: Latin/scientific name, if known.
        summary: Short descriptive summary of the species.
        behaviour: Behavioural notes, if available.
        wikipedia_url: Source Wikipedia article URL, if available.
        images: List of reference images served via the images endpoint.
    """

    common_name: str
    scientific_name: Optional[str]
    summary: str
    behaviour: Optional[str]
    wikipedia_url: Optional[str]
    images: List[ReferenceImage]


def _load_manifest(reference_dir: Path) -> Optional[Dict[str, Any]]:
    """Load and memoize the reference manifest for a directory.

    The result is cached per resolved directory so each request does not
    re-read the file.  A missing or unparseable manifest is cached as ``None``
    (meaning "no references available") so the API degrades gracefully.

    Args:
        reference_dir: Root directory containing ``manifest.json``.

    Returns:
        The parsed manifest dict, or ``None`` when no usable manifest exists.
    """
    key = str(reference_dir.resolve())
    with _cache_lock:
        if key in _manifest_cache:
            return _manifest_cache[key]
        manifest: Optional[Dict[str, Any]] = None
        manifest_path = reference_dir / _MANIFEST_FILENAME
        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                parsed = json.load(handle)
            if isinstance(parsed, dict):
                manifest = parsed
        except (OSError, json.JSONDecodeError):
            manifest = None
        _manifest_cache[key] = manifest
        return manifest


def clear_manifest_cache() -> None:
    """Clear the memoized manifest cache.

    Exposed for tests so a fresh fixture directory is picked up between cases.
    """
    with _cache_lock:
        _manifest_cache.clear()


def _get_species_entry(reference_dir: Path, name: str) -> Dict[str, Any]:
    """Return the manifest entry for a species common name.

    Args:
        reference_dir: Root directory containing the manifest.
        name: Species common name (already URL-decoded by FastAPI).

    Returns:
        The species sub-dict from the manifest.

    Raises:
        HTTPException: 404 if no manifest exists or the species is absent.
    """
    manifest = _load_manifest(reference_dir)
    if manifest is None:
        raise HTTPException(status_code=404, detail="No reference data available")
    species_map = manifest.get("species")
    if not isinstance(species_map, dict) or name not in species_map:
        raise HTTPException(status_code=404, detail="Species reference not found")
    entry = species_map[name]
    if not isinstance(entry, dict):
        raise HTTPException(status_code=404, detail="Species reference not found")
    return entry


@router.get("/{name}/reference", response_model=SpeciesReference)
def get_species_reference(
    name: str,
    reference_dir: Path = Depends(get_reference_dir),
) -> SpeciesReference:
    """Return reference information for a species by common name.

    Args:
        name: Species common name (URL-encoded by the client, decoded here).
        reference_dir: Injected species-reference root directory.

    Returns:
        ``SpeciesReference`` with per-image URLs pointing at the image endpoint.

    Raises:
        HTTPException: 404 if the species has no reference entry.
    """
    entry = _get_species_entry(reference_dir, name)
    raw_images = entry.get("images")
    images: List[ReferenceImage] = []
    if isinstance(raw_images, list):
        encoded_name = quote(name, safe="")
        for index, image in enumerate(raw_images):
            if not isinstance(image, dict):
                continue
            images.append(
                ReferenceImage(
                    url=f"/api/species/{encoded_name}/reference/images/{index}",
                    attribution=str(image.get("attribution", "")),
                    license=image.get("license"),
                )
            )
    return SpeciesReference(
        common_name=str(entry.get("common_name", name)),
        scientific_name=entry.get("scientific_name"),
        summary=str(entry.get("summary", "")),
        behaviour=entry.get("behaviour"),
        wikipedia_url=entry.get("wikipedia_url"),
        images=images,
    )


def _resolve_reference_image(
    reference_dir: Path, name: str, index: int, *, thumb: bool
) -> Path:
    """Resolve the on-disk file for one species reference image or its thumbnail.

    Looks up the manifest image entry, applies the path-traversal guard, and
    returns the file to serve. When ``thumb`` is set the small
    ``thumbnail_path`` rendition is preferred, falling back to the
    full-resolution ``path`` when the entry has no thumbnail or the thumbnail
    file is missing — so a bank built before thumbnails existed still serves
    images, just at full size.

    Args:
        reference_dir: Injected species-reference root directory.
        name: Species common name (decoded from the URL).
        index: Zero-based index into the species' image list.
        thumb: Prefer the thumbnail rendition when one is available.

    Returns:
        The resolved, existing file path beneath ``reference_dir``.

    Raises:
        HTTPException: 404 if the species, index, or file is missing, or if the
            resolved path escapes the reference directory.
    """
    entry = _get_species_entry(reference_dir, name)
    raw_images = entry.get("images")
    if not isinstance(raw_images, list) or not 0 <= index < len(raw_images):
        raise HTTPException(status_code=404, detail="Reference image not found")
    image = raw_images[index]
    if not isinstance(image, dict):
        raise HTTPException(status_code=404, detail="Reference image not found")

    # Prefer the thumbnail (when requested and present), then the full image.
    rel_candidates: List[str] = []
    if thumb and isinstance(image.get("thumbnail_path"), str):
        rel_candidates.append(image["thumbnail_path"])
    if isinstance(image.get("path"), str):
        rel_candidates.append(image["path"])

    base = reference_dir.resolve()
    for rel_path in rel_candidates:
        if not rel_path:
            continue
        candidate = (base / rel_path).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            continue  # Path escapes the bank — treat as absent.
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=404, detail="Reference image not found")


@router.get("/{name}/reference/images/{index}")
def get_species_reference_image(
    name: str,
    index: int,
    reference_dir: Path = Depends(get_reference_dir),
) -> FileResponse:
    """Serve the full-resolution JPEG bytes for one reference image.

    Args:
        name: Species common name (decoded from the URL).
        index: Zero-based index into the species' image list.
        reference_dir: Injected species-reference root directory.

    Returns:
        ``FileResponse`` serving the JPEG with ``image/jpeg`` media type.

    Raises:
        HTTPException: 404 if the species, index, or file is missing, or if the
            resolved path escapes the reference directory.
    """
    candidate = _resolve_reference_image(reference_dir, name, index, thumb=False)
    return FileResponse(str(candidate), media_type="image/jpeg")


@router.get("/{name}/reference/images/{index}/thumbnail")
def get_species_reference_thumbnail(
    name: str,
    index: int,
    reference_dir: Path = Depends(get_reference_dir),
) -> FileResponse:
    """Serve the small square thumbnail for one reference image.

    Falls back to the full-resolution image when the bank has no thumbnail for
    this entry (e.g. it was built before thumbnails existed), so callers can use
    this endpoint unconditionally.

    Args:
        name: Species common name (decoded from the URL).
        index: Zero-based index into the species' image list.
        reference_dir: Injected species-reference root directory.

    Returns:
        ``FileResponse`` serving the JPEG with ``image/jpeg`` media type.

    Raises:
        HTTPException: 404 if the species, index, or file is missing, or if the
            resolved path escapes the reference directory.
    """
    candidate = _resolve_reference_image(reference_dir, name, index, thumb=True)
    return FileResponse(str(candidate), media_type="image/jpeg")
