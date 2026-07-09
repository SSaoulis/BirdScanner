"""Tests for the offline species-reference builder.

These never touch the network: every ``fetch_*`` / ``download_image`` function
is monkeypatched to return canned payloads. Run via ``python -m pytest`` from the
repo root.
"""

import os
import sys

# The builder lives under ``tools/`` (not an installed package), so put the repo
# root on sys.path and import it by module path. This test file lives at
# tests/tools/, so the repo root is three directories up.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# pylint: disable=wrong-import-position
from PIL import Image  # noqa: E402

from tools import build_species_reference as builder  # noqa: E402

# pylint: enable=wrong-import-position


def _write_jpeg(path, size=(240, 160)):
    """Write a small real JPEG at ``path`` so PIL can open/resize it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, (120, 160, 90)).save(path, "JPEG")


# --- slug + pure helpers ---------------------------------------------------


def test_slugify_replaces_non_alphanumeric_runs_with_single_dash():
    """Slugs lowercase and collapse punctuation/spaces into single dashes."""
    assert builder.slugify("Eurasian blue tit") == "eurasian-blue-tit"
    assert builder.slugify("Audouin's gull") == "audouin-s-gull"
    assert builder.slugify("Arabian green bee-eater") == "arabian-green-bee-eater"


def test_resolve_title_uses_override_then_falls_back_to_label():
    """An explicit override title wins; otherwise the label is used verbatim."""
    overrides = {"Audouins gull": {"wikipedia_title": "Audouin's gull"}}
    assert builder.resolve_title("Audouins gull", overrides) == "Audouin's gull"
    assert builder.resolve_title("Arctic tern", overrides) == "Arctic tern"


def test_slice_section_extracts_wiki_heading_body():
    """``_slice_section`` reads a ``== Heading ==`` body up to the next heading."""
    text = (
        "Intro line.\n"
        "== Description ==\n"
        "It is small.\n"
        "== Behaviour ==\n"
        "It forages on the ground.\n"
        "It is territorial.\n"
        "== Breeding ==\n"
        "Lays eggs.\n"
    )
    body = builder._slice_section(text, ("Behaviour",))
    assert body == "It forages on the ground. It is territorial."


def test_slice_section_returns_none_when_no_heading_matches():
    """No matching heading yields ``None`` (best-effort behaviour field)."""
    text = "== Description ==\nSmall bird.\n"
    assert builder._slice_section(text, ("Behaviour",)) is None


# --- canned-fetch fixtures -------------------------------------------------


def _patch_fetches(monkeypatch, *, with_image=True, summary_found=True):
    """Patch all network functions with deterministic canned behaviour."""

    def fake_summary(title):
        if not summary_found:
            return None
        doc = {
            "titles": {"canonical": title.replace(" ", "_")},
            "extract": f"{title} is a small bird.",
            "content_urls": {
                "desktop": {"page": f"https://en.wikipedia.org/wiki/{title}"}
            },
            "wikibase_item": "Q123",
        }
        if with_image:
            doc["originalimage"] = {"source": "https://upload.wikimedia.org/Bird.jpg"}
        return doc

    monkeypatch.setattr(builder, "fetch_wikipedia_summary", fake_summary)
    monkeypatch.setattr(builder, "fetch_scientific_name", lambda qid: "Testus birdus")
    monkeypatch.setattr(builder, "fetch_behaviour", lambda title: "It forages.")
    monkeypatch.setattr(
        builder,
        "fetch_image_metadata",
        lambda t: {"attribution": "A Photographer", "license": "CC BY-SA 4.0"},
    )

    downloaded = []

    def fake_download(source_url, dest_path):
        downloaded.append((source_url, dest_path))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as handle:
            handle.write(b"\xff\xd8\xff")
        return True

    monkeypatch.setattr(builder, "download_image", fake_download)
    return downloaded


# --- build_species_entry ---------------------------------------------------


def test_build_species_entry_shape_and_image(monkeypatch, tmp_path):
    """A found article yields a fully-populated entry with one image on disk."""
    monkeypatch.setattr(builder, "OUTPUT_DIR", str(tmp_path))
    _patch_fetches(monkeypatch)

    entry = builder.build_species_entry("Eurasian blue tit", {})

    assert entry["common_name"] == "Eurasian blue tit"
    assert entry["scientific_name"] == "Testus birdus"
    assert entry["summary"].startswith("Eurasian blue tit")
    assert entry["behaviour"] == "It forages."
    assert entry["wikipedia_url"].startswith("https://en.wikipedia.org/wiki/")
    assert len(entry["images"]) == 1
    image = entry["images"][0]
    assert image["path"] == "images/eurasian-blue-tit/0.jpg"
    assert ".." not in image["path"]
    assert image["attribution"] == "A Photographer"
    assert image["license"] == "CC BY-SA 4.0"
    assert os.path.exists(os.path.join(str(tmp_path), image["path"]))


def test_build_species_entry_override_scientific_name_wins(monkeypatch, tmp_path):
    """An explicit override scientific name overrides the Wikidata value."""
    monkeypatch.setattr(builder, "OUTPUT_DIR", str(tmp_path))
    _patch_fetches(monkeypatch)
    overrides = {"Some bird": {"scientific_name": "Override us"}}

    entry = builder.build_species_entry("Some bird", overrides)

    assert entry["scientific_name"] == "Override us"


def test_build_species_entry_missing_article(monkeypatch, tmp_path):
    """A label with no article yields an empty summary and no images."""
    monkeypatch.setattr(builder, "OUTPUT_DIR", str(tmp_path))
    _patch_fetches(monkeypatch, summary_found=False)

    entry = builder.build_species_entry("Ghost bird", {})

    assert entry["summary"] == ""
    assert entry["images"] == []
    assert entry["wikipedia_url"] is None


# --- build_manifest: overrides, skip, incremental --------------------------


def test_build_manifest_skips_skip_labels(monkeypatch, tmp_path):
    """``skip`` overrides exclude a label from the manifest entirely."""
    monkeypatch.setattr(builder, "OUTPUT_DIR", str(tmp_path))
    _patch_fetches(monkeypatch)
    overrides = {"Unknown": {"skip": True}}

    manifest = builder.build_manifest(
        ["Unknown", "Arctic tern"], overrides, {}, builder.BuildOptions(throttle=0)
    )

    assert "Unknown" not in manifest["species"]
    assert "Arctic tern" in manifest["species"]
    assert manifest["version"] == builder.MANIFEST_VERSION
    assert manifest["source"] == builder.MANIFEST_SOURCE


def test_build_manifest_incremental_skips_complete_entries(monkeypatch, tmp_path):
    """Re-running does not refetch a complete species whose image exists."""
    monkeypatch.setattr(builder, "OUTPUT_DIR", str(tmp_path))
    _patch_fetches(monkeypatch)

    first = builder.build_manifest(
        ["Arctic tern"], {}, {}, builder.BuildOptions(throttle=0)
    )

    calls = []

    def _record_no_summary(title):
        calls.append(title)  # stands in for a missing article (returns None)

    monkeypatch.setattr(builder, "fetch_wikipedia_summary", _record_no_summary)

    second = builder.build_manifest(
        ["Arctic tern"], {}, first, builder.BuildOptions(throttle=0)
    )

    assert not calls  # no refetch
    assert second["species"]["Arctic tern"] == first["species"]["Arctic tern"]


def test_build_manifest_image_only_refresh_when_image_missing(monkeypatch, tmp_path):
    """A missing image is re-downloaded from the cache without any text refetch."""
    monkeypatch.setattr(builder, "OUTPUT_DIR", str(tmp_path))
    _patch_fetches(monkeypatch)
    first = builder.build_manifest(
        ["Arctic tern"], {}, {}, builder.BuildOptions(throttle=0)
    )
    original_entry = dict(first["species"]["Arctic tern"])

    # Delete the downloaded image so the entry is now incomplete.
    img_rel = first["species"]["Arctic tern"]["images"][0]["path"]
    img_path = os.path.join(str(tmp_path), img_rel)
    os.remove(img_path)

    # Track text refetches (must NOT happen) and image downloads (must happen).
    refetched = []

    def _record_summary_call(title):
        refetched.append(title)  # image-only path must never reach this

    monkeypatch.setattr(builder, "fetch_wikipedia_summary", _record_summary_call)
    downloaded = []
    orig_download = builder.download_image

    def tracking_download(source_url, dest_path):
        downloaded.append(source_url)
        return orig_download(source_url, dest_path)

    monkeypatch.setattr(builder, "download_image", tracking_download)

    result = builder.build_manifest(
        ["Arctic tern"], {}, first, builder.BuildOptions(throttle=0)
    )

    assert refetched == []  # no Wikipedia text refetch
    assert downloaded == ["https://upload.wikimedia.org/Bird.jpg"]  # image re-fetched
    assert os.path.exists(img_path)  # original restored on disk
    # Text fields are unchanged; only the image was refreshed.
    refreshed_entry = result["species"]["Arctic tern"]
    for field in ("summary", "scientific_name", "behaviour", "wikipedia_url"):
        assert refreshed_entry[field] == original_entry[field]


def test_build_manifest_force_refetches_everything(monkeypatch, tmp_path):
    """``--force`` refetches even complete entries."""
    monkeypatch.setattr(builder, "OUTPUT_DIR", str(tmp_path))
    _patch_fetches(monkeypatch)
    first = builder.build_manifest(
        ["Arctic tern"], {}, {}, builder.BuildOptions(throttle=0)
    )

    calls = []
    orig = builder.fetch_wikipedia_summary

    def _record_and_delegate(t):
        calls.append(t)
        return orig(t)

    monkeypatch.setattr(builder, "fetch_wikipedia_summary", _record_and_delegate)
    builder.build_manifest(
        ["Arctic tern"], {}, first, builder.BuildOptions(force=True, throttle=0)
    )

    assert calls == ["Arctic tern"]


def test_build_manifest_limit_caps_processed_labels(monkeypatch, tmp_path):
    """``--limit`` processes only the first N labels needing a fetch."""
    monkeypatch.setattr(builder, "OUTPUT_DIR", str(tmp_path))
    _patch_fetches(monkeypatch)

    manifest = builder.build_manifest(
        ["A bird", "B bird", "C bird"],
        {},
        {},
        builder.BuildOptions(limit=2, throttle=0),
    )

    assert len(manifest["species"]) == 2
    assert "A bird" in manifest["species"]
    assert "C bird" not in manifest["species"]


# --- thumbnails ------------------------------------------------------------


def test_thumbnail_rel_path_inserts_suffix():
    """The thumbnail path is the original with a ``_thumb`` suffix."""
    assert (
        builder._thumbnail_rel_path("images/robin/0.jpg") == "images/robin/0_thumb.jpg"
    )


def test_make_thumbnail_produces_square_jpeg(tmp_path):
    """``make_thumbnail`` writes a centre-cropped square JPEG of the given size."""
    src = os.path.join(str(tmp_path), "src.jpg")
    dest = os.path.join(str(tmp_path), "out", "0_thumb.jpg")
    _write_jpeg(src, size=(240, 160))

    assert builder.make_thumbnail(src, dest, size=128) is True
    assert os.path.exists(dest)
    with Image.open(dest) as thumb:
        assert thumb.size == (128, 128)
        assert thumb.format == "JPEG"


def test_make_thumbnail_returns_false_on_bad_source(tmp_path):
    """A corrupt/unreadable source is reported as failure, not an exception."""
    src = os.path.join(str(tmp_path), "bad.jpg")
    with open(src, "wb") as handle:
        handle.write(b"\xff\xd8\xffnotarealjpeg")
    dest = os.path.join(str(tmp_path), "0_thumb.jpg")

    assert builder.make_thumbnail(src, dest) is False
    assert not os.path.exists(dest)


def test_ensure_thumbnails_backfills_missing(tmp_path):
    """A manifest image with an on-disk original but no thumb is backfilled."""
    output_dir = str(tmp_path)
    image_rel = "images/robin/0.jpg"
    _write_jpeg(os.path.join(output_dir, image_rel))
    species_map = {"Robin": {"images": [{"path": image_rel}]}}

    generated = builder.ensure_thumbnails(species_map, output_dir)

    assert generated == 1
    thumb_rel = species_map["Robin"]["images"][0]["thumbnail_path"]
    assert thumb_rel == "images/robin/0_thumb.jpg"
    assert os.path.exists(os.path.join(output_dir, thumb_rel))


def test_ensure_thumbnails_skips_when_thumb_exists(tmp_path):
    """An image with an existing thumbnail file is left untouched."""
    output_dir = str(tmp_path)
    image_rel = "images/robin/0.jpg"
    thumb_rel = "images/robin/0_thumb.jpg"
    _write_jpeg(os.path.join(output_dir, image_rel))
    _write_jpeg(os.path.join(output_dir, thumb_rel), size=(128, 128))
    species_map = {
        "Robin": {"images": [{"path": image_rel, "thumbnail_path": thumb_rel}]}
    }

    assert builder.ensure_thumbnails(species_map, output_dir) == 0


def test_ensure_thumbnails_skips_when_original_missing(tmp_path):
    """No original on disk means no thumbnail is generated."""
    species_map = {"Ghost": {"images": [{"path": "images/ghost/0.jpg"}]}}

    assert builder.ensure_thumbnails(species_map, str(tmp_path)) == 0
    assert "thumbnail_path" not in species_map["Ghost"]["images"][0]


def test_build_species_entry_stamps_thumbnail_path(monkeypatch, tmp_path):
    """When the downloaded image is a real JPEG, the entry gets a thumbnail."""
    monkeypatch.setattr(builder, "OUTPUT_DIR", str(tmp_path))
    _patch_fetches(monkeypatch)

    # Override the canned download to write a real JPEG so make_thumbnail works.
    def real_download(source_url, dest_path):
        _write_jpeg(dest_path)
        return True

    monkeypatch.setattr(builder, "download_image", real_download)

    entry = builder.build_species_entry("Eurasian blue tit", {})

    image = entry["images"][0]
    assert image["thumbnail_path"] == "images/eurasian-blue-tit/0_thumb.jpg"
    assert os.path.exists(os.path.join(str(tmp_path), image["thumbnail_path"]))


# --- coverage report -------------------------------------------------------


def test_compute_coverage_populates_all_buckets(monkeypatch):
    """Coverage report buckets missing / no-image / no-sci-name / skipped."""
    species = {
        "Good bird": {
            "summary": "x",
            "images": [{"path": "images/good/0.jpg"}],
            "scientific_name": "Goodus",
        },
        "No image bird": {
            "summary": "x",
            "images": [],
            "scientific_name": "Noimagus",
        },
        "No sci bird": {
            "summary": "x",
            "images": [{"path": "images/nosci/0.jpg"}],
            "scientific_name": None,
        },
        "Missing bird": {"summary": "", "images": [], "scientific_name": None},
    }
    overrides = {"Skip bird": {"skip": True}}
    labels = ["Good bird", "No image bird", "No sci bird", "Missing bird", "Skip bird"]

    report = builder.compute_coverage(species, overrides, labels)

    assert report["skipped"] == ["Skip bird"]
    assert "Missing bird" in report["missing"]
    assert set(report["no_images"]) == {"No image bird", "Missing bird"}
    assert set(report["no_scientific_name"]) == {"No sci bird", "Missing bird"}
    assert "Good bird" not in report["missing"]
