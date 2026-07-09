"""Integration tests for the read-only species-reference API.

All tests build a temporary ``species_reference`` directory with a fixture
``manifest.json`` and a fake JPEG, then override ``get_reference_dir`` to point
there so no real data bank is required.
"""

import json
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from birdscanner.api.main import app
from birdscanner.api.dependencies import get_reference_dir
from birdscanner.api.routers import reference

_SPECIES_NAME = "Eurasian Blue Tit"
# A second species whose image has no cached thumbnail, to exercise the
# thumbnail endpoint's fallback to the full-resolution image.
_NO_THUMB_NAME = "Common Chaffinch"
_FAKE_JPEG = b"\xff\xd8\xffFAKEJPEGDATA"
_FAKE_THUMB = b"\xff\xd8\xffFAKETHUMBNAIL"


def _write_manifest(reference_dir: Path) -> None:
    """Write a fixture manifest plus fake JPEGs into ``reference_dir``.

    The blue tit entry carries both a full image and a cached ``thumbnail_path``;
    the chaffinch entry has only a full image (no thumbnail) so tests can check
    the thumbnail endpoint falls back to it.
    """
    image_rel = "images/eurasian-blue-tit/0.jpg"
    thumb_rel = "images/eurasian-blue-tit/0_thumb.jpg"
    image_path = reference_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(_FAKE_JPEG)
    (reference_dir / thumb_rel).write_bytes(_FAKE_THUMB)

    no_thumb_rel = "images/common-chaffinch/0.jpg"
    no_thumb_path = reference_dir / no_thumb_rel
    no_thumb_path.parent.mkdir(parents=True, exist_ok=True)
    no_thumb_path.write_bytes(_FAKE_JPEG)

    manifest = {
        "version": 1,
        "generated_at": "2026-06-17T00:00:00Z",
        "source": "wikipedia+wikidata",
        "species": {
            _SPECIES_NAME: {
                "common_name": _SPECIES_NAME,
                "scientific_name": "Cyanistes caeruleus",
                "summary": "A small, colourful passerine bird.",
                "behaviour": "Acrobatic feeder, often hangs upside down.",
                "wikipedia_url": "https://en.wikipedia.org/wiki/Eurasian_blue_tit",
                "images": [
                    {
                        "path": image_rel,
                        "thumbnail_path": thumb_rel,
                        "source_url": "https://example.com/original.jpg",
                        "attribution": "Photo by Someone",
                        "license": "CC BY-SA 4.0",
                    }
                ],
            },
            _NO_THUMB_NAME: {
                "common_name": _NO_THUMB_NAME,
                "scientific_name": "Fringilla coelebs",
                "summary": "A common finch.",
                "behaviour": None,
                "wikipedia_url": None,
                "images": [
                    {
                        "path": no_thumb_rel,
                        "source_url": "https://example.com/chaffinch.jpg",
                        "attribution": "Photo by Someone Else",
                        "license": None,
                    }
                ],
            },
        },
    }
    (reference_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture()
def reference_dir(tmp_path: Path) -> Path:
    """Temporary reference directory populated with a fixture manifest."""
    ref_dir = tmp_path / "species_reference"
    ref_dir.mkdir()
    _write_manifest(ref_dir)
    return ref_dir


@pytest.fixture()
def empty_reference_dir(tmp_path: Path) -> Path:
    """Temporary reference directory with no manifest on disk."""
    ref_dir = tmp_path / "empty_reference"
    ref_dir.mkdir()
    return ref_dir


@pytest.fixture()
def client(reference_dir: Path):
    """TestClient with the reference directory dependency overridden."""
    reference.clear_manifest_cache()
    app.dependency_overrides[get_reference_dir] = lambda: reference_dir
    yield TestClient(app)
    app.dependency_overrides.clear()
    reference.clear_manifest_cache()


@pytest.fixture()
def empty_client(empty_reference_dir: Path):
    """TestClient pointed at a directory with no manifest."""
    reference.clear_manifest_cache()
    app.dependency_overrides[get_reference_dir] = lambda: empty_reference_dir
    yield TestClient(app)
    app.dependency_overrides.clear()
    reference.clear_manifest_cache()


class TestSpeciesReference:
    def test_returns_expected_shape(self, client):
        encoded = quote(_SPECIES_NAME, safe="")
        resp = client.get(f"/api/species/{encoded}/reference")
        assert resp.status_code == 200
        data = resp.json()
        assert data["common_name"] == _SPECIES_NAME
        assert data["scientific_name"] == "Cyanistes caeruleus"
        assert data["summary"].startswith("A small")
        assert data["behaviour"]
        assert data["wikipedia_url"].startswith("https://")
        assert len(data["images"]) == 1
        image = data["images"][0]
        assert image["attribution"] == "Photo by Someone"
        assert image["license"] == "CC BY-SA 4.0"
        assert image["url"] == f"/api/species/{encoded}/reference/images/0"

    def test_path_and_source_url_not_exposed(self, client):
        encoded = quote(_SPECIES_NAME, safe="")
        resp = client.get(f"/api/species/{encoded}/reference")
        image = resp.json()["images"][0]
        assert "path" not in image
        assert "source_url" not in image

    def test_unknown_species_404(self, client):
        resp = client.get("/api/species/Dodo/reference")
        assert resp.status_code == 404
        assert "detail" in resp.json()


class TestReferenceImage:
    def test_serves_jpeg_bytes(self, client):
        encoded = quote(_SPECIES_NAME, safe="")
        resp = client.get(f"/api/species/{encoded}/reference/images/0")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == _FAKE_JPEG

    def test_out_of_range_index_404(self, client):
        encoded = quote(_SPECIES_NAME, safe="")
        resp = client.get(f"/api/species/{encoded}/reference/images/5")
        assert resp.status_code == 404

    def test_unknown_species_image_404(self, client):
        resp = client.get("/api/species/Dodo/reference/images/0")
        assert resp.status_code == 404

    def test_path_traversal_rejected(self, client, reference_dir: Path):
        # Plant a manifest entry whose path escapes the reference directory.
        outside = reference_dir.parent / "secret.jpg"
        outside.write_bytes(b"TOPSECRET")
        manifest_path = reference_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["species"]["Evil"] = {
            "common_name": "Evil",
            "scientific_name": None,
            "summary": "x",
            "behaviour": None,
            "wikipedia_url": None,
            "images": [
                {
                    "path": "../secret.jpg",
                    "source_url": "",
                    "attribution": "",
                    "license": None,
                }
            ],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        reference.clear_manifest_cache()

        resp = client.get("/api/species/Evil/reference/images/0")
        assert resp.status_code == 404


class TestReferenceThumbnail:
    def test_serves_thumbnail_when_present(self, client):
        encoded = quote(_SPECIES_NAME, safe="")
        resp = client.get(f"/api/species/{encoded}/reference/images/0/thumbnail")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == _FAKE_THUMB

    def test_falls_back_to_full_image_when_no_thumbnail(self, client):
        encoded = quote(_NO_THUMB_NAME, safe="")
        resp = client.get(f"/api/species/{encoded}/reference/images/0/thumbnail")
        assert resp.status_code == 200
        # The entry has no thumbnail, so the full image is served instead.
        assert resp.content == _FAKE_JPEG

    def test_out_of_range_index_404(self, client):
        encoded = quote(_SPECIES_NAME, safe="")
        resp = client.get(f"/api/species/{encoded}/reference/images/5/thumbnail")
        assert resp.status_code == 404

    def test_unknown_species_404(self, client):
        resp = client.get("/api/species/Dodo/reference/images/0/thumbnail")
        assert resp.status_code == 404


class TestMissingManifest:
    def test_reference_404_clean(self, empty_client):
        encoded = quote(_SPECIES_NAME, safe="")
        resp = empty_client.get(f"/api/species/{encoded}/reference")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_image_404_clean(self, empty_client):
        encoded = quote(_SPECIES_NAME, safe="")
        resp = empty_client.get(f"/api/species/{encoded}/reference/images/0")
        assert resp.status_code == 404
