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
_FAKE_JPEG = b"\xff\xd8\xffFAKEJPEGDATA"


def _write_manifest(reference_dir: Path) -> None:
    """Write a fixture manifest plus a fake JPEG into ``reference_dir``."""
    image_rel = "images/eurasian-blue-tit/0.jpg"
    image_path = reference_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(_FAKE_JPEG)

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
                        "source_url": "https://example.com/original.jpg",
                        "attribution": "Photo by Someone",
                        "license": "CC BY-SA 4.0",
                    }
                ],
            }
        },
    }
    (reference_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


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
