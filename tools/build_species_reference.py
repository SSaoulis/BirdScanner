#!/usr/bin/env python3
"""Offline build-time tool: fetch reference info for every bird species label.

This script is **not** part of the BirdScanner runtime. Run it on a dev machine
with internet access to build a cached reference manifest plus locally
downloaded reference images for every species the classifier can predict.

Data sources
------------
- Wikipedia REST API (``/api/rest_v1/page/summary/<Title>``) for the article
  summary, lead image, canonical URL, and the Wikidata entity id.
- MediaWiki API (``/w/api.php``) for a "Behaviour"/"Habitat" section's plain
  text and for image metadata (attribution/license via ``imageinfo`` +
  ``extmetadata``).
- Wikidata (``/wiki/Special:EntityData/<id>.json``) for the scientific name
  (taxon name, property ``P225``).

Source of truth for the species list is
``assets/models/convnext_v2_tiny.onnx_class_to_idx.json`` (a ``{name: index}`` map).
The label ``"Unknown"`` is a non-bird sentinel and is skipped.

Outputs (all under ``assets/species_reference/``)
-------------------------------------------------
- ``manifest.json`` — the cached reference data (see ``--help`` for schema).
- ``images/<slug>/<n>.jpg`` — downloaded reference images, each with a small
  square ``<n>_thumb.jpg`` sibling (``thumbnail_path`` in the manifest).
- ``coverage_report.json`` — labels with no article / no images / no scientific
  name / skipped.
- ``overrides.json`` — hand-maintainable name resolution + value overrides.

Incremental
-----------
Re-running only fills in labels missing from the manifest (or whose image files
are missing on disk); already-complete species are skipped. When only an image
file is missing but the cached text (summary/scientific name/behaviour) is
intact, that species takes a fast **image-only** path: the thumbnail is
re-downloaded from the manifest's stored ``source_url`` with **no** Wikipedia /
Wikidata text calls (one request instead of ~six). A ``tqdm`` progress bar
tracks the species being processed. Use ``--force`` to refetch everything and
``--limit N`` to process only the first N missing labels.

Full build (one-time, run by a human; do not run in CI):

    python tools/build_species_reference.py

Quick validation against the live network:

    python tools/build_species_reference.py --limit 5
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from tqdm import tqdm  # dev-only tool; progress bar for the long build

from tools.reference_thumbnails import (
    ensure_thumbnails,
    make_thumbnail,
    _thumbnail_rel_path,
)

# --- Paths -----------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
CLASS_TO_IDX_PATH = os.path.join(
    REPO_ROOT, "assets", "models", "convnext_v2_tiny.onnx_class_to_idx.json"
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "assets", "species_reference")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")
OVERRIDES_PATH = os.path.join(OUTPUT_DIR, "overrides.json")
COVERAGE_PATH = os.path.join(OUTPUT_DIR, "coverage_report.json")
IMAGES_SUBDIR = "images"

# --- Constants -------------------------------------------------------------

USER_AGENT = (
    "BirdScanner-ReferenceBuilder/1.0 "
    "(https://github.com/stefansaoulis/BirdFinder; contact: stefan.sooley@gmail.com)"
)
WIKIPEDIA_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_ENTITY_API = "https://www.wikidata.org/wiki/Special:EntityData/"
MANIFEST_VERSION = 1
MANIFEST_SOURCE = "wikipedia+wikidata"
REQUEST_TIMEOUT = 30
THROTTLE_SECONDS = 0.5
# Section titles we treat as the "behaviour" field, in priority order.
BEHAVIOUR_SECTION_TITLES = (
    "Behaviour",
    "Behavior",
    "Behaviour and ecology",
    "Behavior and ecology",
    "Ecology",
    "Habitat",
    "Habitat and distribution",
    "Distribution and habitat",
)

# Default overrides seeded into a fresh overrides.json. The sentinel "Unknown"
# label is always skipped; the apostrophe-stripped labels below are the obvious
# resolution failures found by spot-checking the label list.
DEFAULT_OVERRIDES: dict[str, dict[str, Any]] = {
    "Unknown": {"skip": True},
    "Audouins gull": {"wikipedia_title": "Audouin's gull"},
    "Bonellis eagle": {"wikipedia_title": "Bonelli's eagle"},
    "Cettis warbler": {"wikipedia_title": "Cetti's warbler"},
    "Sykess warbler": {"wikipedia_title": "Sykes's warbler"},
    "Pallass leaf warbler": {"wikipedia_title": "Pallas's leaf warbler"},
    "Pallass grasshopper warbler": {"wikipedia_title": "Pallas's grasshopper warbler"},
    "Pallass reed bunting": {"wikipedia_title": "Pallas's reed bunting"},
    "Pallass gull": {"wikipedia_title": "Pallas's gull"},
    "Temmincks stint": {"wikipedia_title": "Temminck's stint"},
    "Montagus harrier": {"wikipedia_title": "Montagu's harrier"},
}


# --- Generic HTTP helpers --------------------------------------------------


def _http_get(url: str) -> bytes:
    """Perform an HTTP GET with the project User-Agent and return raw bytes.

    Args:
        url: Fully-qualified URL to fetch.

    Returns:
        The response body as bytes.

    Raises:
        urllib.error.URLError / HTTPError on network or HTTP failure.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return response.read()


def _http_get_json(url: str) -> Any:
    """Fetch a URL and parse the response body as JSON.

    Args:
        url: Fully-qualified URL returning a JSON body.

    Returns:
        The decoded JSON document (dict/list/scalar).
    """
    return json.loads(_http_get(url).decode("utf-8"))


# --- Pure helpers ----------------------------------------------------------


def slugify(label: str) -> str:
    """Convert a species label into a filesystem-safe slug.

    Lowercases the label and replaces every run of non-alphanumeric characters
    with a single dash, trimming leading/trailing dashes.

    Args:
        label: The species common name exactly as in ``class_to_idx``.

    Returns:
        A slug such as ``"eurasian-blue-tit"``.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "unknown"


def load_class_labels(path: str = CLASS_TO_IDX_PATH) -> list[str]:
    """Load the ordered list of species labels from the class-to-index map.

    Args:
        path: Path to the ``convnext_v2_tiny.onnx_class_to_idx.json`` file.

    Returns:
        Labels sorted by their integer class index.
    """
    with open(path, "r", encoding="utf-8") as handle:
        mapping: dict[str, int] = json.load(handle)
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def resolve_title(label: str, overrides: dict[str, dict[str, Any]]) -> str:
    """Resolve a species label to the Wikipedia article title to query.

    Consults ``overrides`` first (an explicit ``wikipedia_title`` wins),
    otherwise uses the label verbatim.

    Args:
        label: The species common name.
        overrides: The loaded overrides map.

    Returns:
        The Wikipedia article title to look up.
    """
    override = overrides.get(label, {})
    title = override.get("wikipedia_title")
    return title if isinstance(title, str) and title else label


# --- Network fetch functions (mocked in tests) -----------------------------


def fetch_wikipedia_summary(title: str) -> Optional[dict[str, Any]]:
    """Fetch the Wikipedia REST summary for an article title.

    Args:
        title: The Wikipedia article title (spaces or underscores both work).

    Returns:
        The parsed summary document, or ``None`` if the article does not exist
        / is a disambiguation page / the request fails.
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = WIKIPEDIA_SUMMARY_API + encoded
    try:
        data = _http_get_json(url)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("type") == "disambiguation":
        return None
    if "extract" not in data and "title" not in data:
        return None
    return data


def fetch_behaviour(title: str) -> Optional[str]:
    """Fetch the plain text of a behaviour/habitat section for an article.

    Walks the article's section list (MediaWiki ``parse`` with
    ``prop=sections``), finds the first section whose title matches one of
    ``BEHAVIOUR_SECTION_TITLES``, then fetches that section's plain extract.

    Args:
        title: The Wikipedia article title.

    Returns:
        The section's plain text, or ``None`` when no such section exists.
    """
    sections_url = (
        MEDIAWIKI_API
        + "?"
        + urllib.parse.urlencode(
            {
                "action": "parse",
                "page": title,
                "prop": "sections",
                "format": "json",
                "redirects": "1",
            }
        )
    )
    try:
        sections_doc = _http_get_json(sections_url)
        sections = sections_doc.get("parse", {}).get("sections", [])
    except Exception:
        return None

    section_index: Optional[str] = None
    for title_lower in (t.lower() for t in BEHAVIOUR_SECTION_TITLES):
        for section in sections:
            if section.get("line", "").lower() == title_lower:
                section_index = section.get("index")
                break
        if section_index:
            break
    if not section_index:
        return None

    extract_url = (
        MEDIAWIKI_API
        + "?"
        + urllib.parse.urlencode(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "titles": title,
                "format": "json",
                "redirects": "1",
            }
        )
    )
    try:
        extract_doc = _http_get_json(extract_url)
        pages = extract_doc.get("query", {}).get("pages", {})
    except Exception:
        return None

    for page in pages.values():
        full_text = page.get("extract", "")
        section_text = _slice_section(full_text, BEHAVIOUR_SECTION_TITLES)
        if section_text:
            return section_text
    return None


_HEADING_RE = re.compile(r"^(=+)\s*(.*?)\s*=+\s*$")


def _slice_section(full_text: str, section_titles: tuple[str, ...]) -> Optional[str]:
    """Extract a single section's body from a plain-text article extract.

    The MediaWiki plaintext extract renders headings with wiki markup, e.g.
    ``"== Behaviour and ecology =="``. This finds the first matching heading and
    returns the text up to the next heading of equal-or-shallower depth.

    Args:
        full_text: The full plain-text article extract.
        section_titles: Candidate heading titles to match (case-insensitive).

    Returns:
        The trimmed section body, or ``None`` if no heading matched.
    """
    wanted = {t.lower() for t in section_titles}
    collected: list[str] = []
    capturing = False
    capture_depth = 0
    for line in full_text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            depth = len(match.group(1))
            heading = match.group(2).strip().lower()
            if capturing and depth <= capture_depth:
                break
            if not capturing and heading in wanted:
                capturing = True
                capture_depth = depth
                continue
        elif capturing and line.strip():
            collected.append(line.strip())
    text = " ".join(collected).strip()
    return text or None


def fetch_scientific_name(wikidata_id: str) -> Optional[str]:
    """Fetch the scientific (taxon) name for a Wikidata entity.

    Reads property ``P225`` (taxon name) from the entity document.

    Args:
        wikidata_id: A Wikidata entity id such as ``"Q25457"``.

    Returns:
        The taxon name string, or ``None`` if unavailable.
    """
    url = WIKIDATA_ENTITY_API + urllib.parse.quote(wikidata_id, safe="") + ".json"
    try:
        doc = _http_get_json(url)
        entity = doc.get("entities", {}).get(wikidata_id, {})
        claims = entity.get("claims", {})
        p225 = claims.get("P225", [])
        value = p225[0]["mainsnak"]["datavalue"]["value"]
    except Exception:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        return text if isinstance(text, str) else None
    return None


def fetch_image_metadata(image_title: str) -> dict[str, Optional[str]]:
    """Fetch attribution/license metadata for a Wikimedia image file.

    Uses the MediaWiki ``imageinfo`` query with ``extmetadata`` to read the
    artist/attribution and license short name.

    Args:
        image_title: The image file title (e.g. ``"File:Blue tit.jpg"``).

    Returns:
        A dict with ``attribution`` and ``license`` keys (values may be None).
    """
    url = (
        MEDIAWIKI_API
        + "?"
        + urllib.parse.urlencode(
            {
                "action": "query",
                "titles": image_title,
                "prop": "imageinfo",
                "iiprop": "extmetadata|url",
                "format": "json",
            }
        )
    )
    result: dict[str, Optional[str]] = {"attribution": None, "license": None}
    try:
        doc = _http_get_json(url)
        pages = doc.get("query", {}).get("pages", {})
    except Exception:
        return result
    for page in pages.values():
        infos = page.get("imageinfo", [])
        if not infos:
            continue
        meta = infos[0].get("extmetadata", {})
        artist = meta.get("Artist", {}).get("value")
        if artist:
            result["attribution"] = _strip_html(artist)
        license_short = meta.get("LicenseShortName", {}).get("value")
        if license_short:
            result["license"] = _strip_html(license_short)
    return result


def _strip_html(value: str) -> str:
    """Strip HTML tags and collapse whitespace in a metadata string.

    Args:
        value: A possibly-HTML metadata string from ``extmetadata``.

    Returns:
        Plain text with tags removed and whitespace collapsed.
    """
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", text).strip()


def download_image(source_url: str, dest_path: str) -> bool:
    """Download an image to a destination path, creating parent dirs.

    Args:
        source_url: The image URL to download.
        dest_path: Absolute destination path on disk.

    Returns:
        ``True`` on success, ``False`` if the download failed.
    """
    try:
        data = _http_get(source_url)
    except Exception:
        return False
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as handle:
        handle.write(data)
    return True


# --- Manifest / overrides persistence --------------------------------------


def load_json_file(path: str, default: Any) -> Any:
    """Load a JSON file, returning ``default`` if it does not exist.

    Args:
        path: Path to the JSON file.
        default: Value to return when the file is absent.

    Returns:
        The parsed JSON document, or ``default``.
    """
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path: str, payload: Any) -> None:
    """Write a JSON document to disk with indentation, creating parent dirs.

    Args:
        path: Destination path.
        payload: JSON-serialisable object to write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def ensure_overrides_seed(path: str = OVERRIDES_PATH) -> dict[str, dict[str, Any]]:
    """Load the overrides file, seeding it with defaults if it does not exist.

    Args:
        path: Path to ``overrides.json``.

    Returns:
        The loaded (or freshly-seeded) overrides map.
    """
    if not os.path.exists(path):
        write_json_file(path, DEFAULT_OVERRIDES)
        return dict(DEFAULT_OVERRIDES)
    loaded = load_json_file(path, {})
    return loaded if isinstance(loaded, dict) else {}


def species_is_complete(entry: dict[str, Any]) -> bool:
    """Return whether a manifest species entry is complete on disk.

    An entry is complete if it has a summary and every listed image file exists.
    Entries with zero images are still considered complete (the article simply
    had no lead image) so we don't refetch them every run.

    Args:
        entry: A manifest species entry.

    Returns:
        ``True`` if no refetch is needed.
    """
    if not entry.get("summary"):
        return False
    for image in entry.get("images", []):
        abs_path = os.path.join(OUTPUT_DIR, image.get("path", ""))
        if not os.path.exists(abs_path):
            return False
    return True


# --- Per-species build -----------------------------------------------------


def build_species_entry(
    label: str,
    overrides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Fetch and assemble the full reference entry for one species label.

    Performs all network I/O via the thin ``fetch_*`` / ``download_image``
    helpers (mocked in tests) and downloads the lead image to
    ``images/<slug>/0.jpg``.

    Args:
        label: The species common name (key in the manifest).
        overrides: The loaded overrides map (explicit values win over fetched).

    Returns:
        A manifest species entry dict (always includes ``common_name``;
        ``summary`` may be empty when the article was not found).
    """
    override = overrides.get(label, {})
    title = resolve_title(label, overrides)
    slug = slugify(label)

    summary_doc = fetch_wikipedia_summary(title)
    entry: dict[str, Any] = {
        "common_name": label,
        "scientific_name": None,
        "summary": "",
        "behaviour": None,
        "wikipedia_url": None,
        "images": [],
    }

    if summary_doc is None:
        # Apply explicit override scientific name even without an article.
        if isinstance(override.get("scientific_name"), str):
            entry["scientific_name"] = override["scientific_name"]
        return entry

    canonical_title = summary_doc.get("titles", {}).get("canonical", title)
    entry["summary"] = summary_doc.get("extract", "") or ""
    entry["wikipedia_url"] = (
        summary_doc.get("content_urls", {}).get("desktop", {}).get("page")
    )

    # Scientific name: override wins, else Wikidata P225.
    if isinstance(override.get("scientific_name"), str):
        entry["scientific_name"] = override["scientific_name"]
    else:
        wikidata_id = summary_doc.get("wikibase_item")
        if isinstance(wikidata_id, str) and wikidata_id:
            entry["scientific_name"] = fetch_scientific_name(wikidata_id)

    entry["behaviour"] = fetch_behaviour(canonical_title)

    # Lead image -> images/<slug>/0.jpg (+ a 0_thumb.jpg thumbnail sibling).
    image_source = (summary_doc.get("originalimage") or {}).get("source") or (
        summary_doc.get("thumbnail") or {}
    ).get("source")
    image_entry = _download_lead_image(image_source, slug) if image_source else None
    if image_entry is not None:
        entry["images"].append(image_entry)
    return entry


def _download_lead_image(image_source: str, slug: str) -> Optional[dict[str, Any]]:
    """Download a species' lead image + thumbnail and build its manifest entry.

    Args:
        image_source: The upstream image URL to download.
        slug: The species slug used for the on-disk ``images/<slug>/`` folder.

    Returns:
        The manifest image dict (``path``, ``source_url``, ``attribution``,
        ``license``, and ``thumbnail_path`` when a thumbnail was written), or
        ``None`` when the download failed.
    """
    rel_path = f"{IMAGES_SUBDIR}/{slug}/0.jpg"
    dest_path = os.path.join(OUTPUT_DIR, rel_path)
    if not download_image(image_source, dest_path):
        return None
    file_title = _file_title_from_url(image_source)
    meta = (
        fetch_image_metadata(file_title)
        if file_title
        else {"attribution": None, "license": None}
    )
    image_entry: dict[str, Any] = {
        "path": rel_path,
        "source_url": image_source,
        "attribution": meta.get("attribution"),
        "license": meta.get("license"),
    }
    thumb_rel = _thumbnail_rel_path(rel_path)
    if make_thumbnail(dest_path, os.path.join(OUTPUT_DIR, thumb_rel)):
        image_entry["thumbnail_path"] = thumb_rel
    return image_entry


def _file_title_from_url(image_url: str) -> Optional[str]:
    """Derive a ``File:`` title from a Wikimedia upload URL.

    Args:
        image_url: An upload.wikimedia.org image URL.

    Returns:
        A ``"File:<name>"`` title, or ``None`` if it cannot be derived.
    """
    name = os.path.basename(urllib.parse.urlparse(image_url).path)
    if not name:
        return None
    return "File:" + urllib.parse.unquote(name)


# --- Image-only incremental refresh ----------------------------------------


def _missing_image_entries(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the entry's image dicts whose original file is missing on disk.

    Args:
        entry: A manifest species entry.

    Returns:
        The subset of ``entry["images"]`` whose ``path`` is absent under
        ``OUTPUT_DIR`` (empty when every original is present).
    """
    missing: list[dict[str, Any]] = []
    for image in entry.get("images", []):
        rel_path = image.get("path", "")
        if rel_path and not os.path.exists(os.path.join(OUTPUT_DIR, rel_path)):
            missing.append(image)
    return missing


def can_refresh_images_only(entry: dict[str, Any]) -> bool:
    """Whether an incomplete entry can be fixed by re-downloading images alone.

    The fast path for a "just fetch the thumbnails" re-run: holds when the cached
    text is intact (non-empty ``summary``) and every missing image file still
    carries a ``source_url``, so no Wikipedia/Wikidata text call is needed.

    Args:
        entry: A manifest species entry (assumed already known incomplete).

    Returns:
        ``True`` when :func:`refresh_species_images` alone can restore the entry.
    """
    if not entry.get("summary"):
        return False
    missing = _missing_image_entries(entry)
    if not missing:
        return False
    return all(
        isinstance(image.get("source_url"), str) and image.get("source_url")
        for image in missing
    )


def refresh_species_images(entry: dict[str, Any]) -> dict[str, Any]:
    """Re-download an entry's missing image files (and thumbnails) in place.

    Reuses each image's cached ``source_url``/attribution/license and every text
    field — no Wikipedia/Wikidata calls. Mirrors the download+thumbnail step of
    :func:`_download_lead_image`; a failed download/thumbnail is skipped.

    Args:
        entry: An entry eligible per :func:`can_refresh_images_only`.

    Returns:
        The same ``entry``, mutated with any regenerated ``thumbnail_path``.
    """
    for image in _missing_image_entries(entry):
        source_url = image.get("source_url")
        rel_path = image.get("path", "")
        if not source_url or not rel_path:
            continue
        dest_path = os.path.join(OUTPUT_DIR, rel_path)
        if not download_image(source_url, dest_path):
            continue
        thumb_rel = _thumbnail_rel_path(rel_path)
        if make_thumbnail(dest_path, os.path.join(OUTPUT_DIR, thumb_rel)):
            image["thumbnail_path"] = thumb_rel
    return entry


# --- Coverage / orchestration ----------------------------------------------


def compute_coverage(
    manifest_species: dict[str, Any],
    overrides: dict[str, dict[str, Any]],
    all_labels: list[str],
) -> dict[str, list[str]]:
    """Compute the coverage report from the manifest and overrides.

    Args:
        manifest_species: The manifest ``species`` map.
        overrides: The loaded overrides map.
        all_labels: Every label in the class-to-index map.

    Returns:
        A dict with ``missing``, ``no_images``, ``no_scientific_name`` and
        ``skipped`` lists.
    """
    report: dict[str, list[str]] = {
        "missing": [],
        "no_images": [],
        "no_scientific_name": [],
        "skipped": [],
    }
    for label in all_labels:
        if overrides.get(label, {}).get("skip"):
            report["skipped"].append(label)
            continue
        entry = manifest_species.get(label)
        if entry is None:
            continue
        if not entry.get("summary"):
            report["missing"].append(label)
        if not entry.get("images"):
            report["no_images"].append(label)
        if not entry.get("scientific_name"):
            report["no_scientific_name"].append(label)
    return report


@dataclass(frozen=True)
class BuildOptions:
    """Options controlling an incremental manifest build.

    Attributes:
        force: Refetch every non-skipped label, ignoring the cache.
        limit: Process at most this many labels that need fetching (``None`` for
            no limit).
        throttle: Seconds to sleep between species (politeness).
    """

    force: bool = False
    limit: Optional[int] = None
    throttle: float = THROTTLE_SECONDS


def _plan_todo(
    labels: list[str],
    overrides: dict[str, dict[str, Any]],
    species: dict[str, Any],
    options: BuildOptions,
) -> list[tuple[str, str]]:
    """Classify labels into the ordered ``(label, mode)`` work list for a build.

    ``mode`` is ``"image"`` when the label only needs its missing images
    re-downloaded (see :func:`can_refresh_images_only`), else ``"full"``. Skipped
    and already-complete labels are omitted; ``force`` makes every label
    ``"full"``; ``limit`` caps the list to the first N labels needing work.

    Args:
        labels: All species labels (sorted by class index).
        overrides: The loaded overrides map.
        species: The working species map (skip labels already removed).
        options: Build controls; see :class:`BuildOptions`.

    Returns:
        The ``(label, mode)`` pairs to process, in label order.
    """
    todo: list[tuple[str, str]] = []
    for label in labels:
        if overrides.get(label, {}).get("skip"):
            continue
        entry = species.get(label)
        if not options.force and entry is not None and species_is_complete(entry):
            continue
        mode = (
            "image"
            if not options.force
            and entry is not None
            and can_refresh_images_only(entry)
            else "full"
        )
        todo.append((label, mode))
        if options.limit is not None and len(todo) >= options.limit:
            break
    return todo


def build_manifest(
    labels: list[str],
    overrides: dict[str, dict[str, Any]],
    existing: dict[str, Any],
    options: BuildOptions = BuildOptions(),
) -> dict[str, Any]:
    """Build (or incrementally update) the species reference manifest.

    Args:
        labels: All species labels (sorted by class index).
        overrides: The loaded overrides map.
        existing: The previously-written manifest (``{}`` for a fresh build).
        options: Build controls (force/limit/throttle); see :class:`BuildOptions`.

    Returns:
        The complete manifest document ready to serialise.
    """
    species: dict[str, Any] = dict(existing.get("species", {}))
    for label in labels:
        if overrides.get(label, {}).get("skip"):
            species.pop(label, None)

    todo = _plan_todo(labels, overrides, species, options)
    for label, mode in tqdm(todo, desc="Species reference", unit="species"):
        if mode == "image":
            refresh_species_images(species[label])
        else:
            species[label] = build_species_entry(label, overrides)
        if options.throttle:
            time.sleep(options.throttle)

    return {
        "version": MANIFEST_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": MANIFEST_SOURCE,
        "species": species,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the offline species reference manifest + images for "
            "BirdScanner. Incremental by default; only missing labels are "
            "fetched. Run the full build (all 707 species) with no flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch every species, ignoring the existing manifest/cache.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only process the first N labels that need fetching (validation).",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=THROTTLE_SECONDS,
        metavar="SECONDS",
        help=f"Seconds to sleep between species (default {THROTTLE_SECONDS}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point: build the manifest, download images, write reports.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).
    """
    args = parse_args(argv)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    labels = load_class_labels()
    overrides = ensure_overrides_seed()
    existing = load_json_file(MANIFEST_PATH, {})

    manifest = build_manifest(
        labels,
        overrides,
        existing,
        BuildOptions(force=args.force, limit=args.limit, throttle=args.throttle),
    )
    # Backfill thumbnails for any originals that lack one (cheap local resize),
    # so a re-run against an existing bank gains thumbnails without re-fetching.
    thumbs = ensure_thumbnails(manifest["species"], OUTPUT_DIR)
    write_json_file(MANIFEST_PATH, manifest)

    coverage = compute_coverage(manifest["species"], overrides, labels)
    write_json_file(COVERAGE_PATH, coverage)

    print(f"Wrote {MANIFEST_PATH} ({len(manifest['species'])} species).")
    print(f"Generated {thumbs} reference thumbnails.")
    print(
        "Coverage: "
        f"{len(coverage['missing'])} missing, "
        f"{len(coverage['no_images'])} without images, "
        f"{len(coverage['no_scientific_name'])} without scientific name, "
        f"{len(coverage['skipped'])} skipped."
    )


if __name__ == "__main__":
    main()
