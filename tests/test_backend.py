"""Integration tests for the FastAPI backend.

All tests use an in-memory SQLite database and a temporary image directory so
no external dependencies are required.
"""

import io
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from db.database import make_session_factory
from db.models import DetectionRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine(tmp_path):
    """In-memory SQLite engine shared across threads via StaticPool."""
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session_factory(engine):
    """Session factory bound to the test engine."""
    return make_session_factory(engine)


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Temporary directory that acts as IMAGE_DIR."""
    return tmp_path


def _make_record(
    session: Session,
    image_dir: Path,
    species: str = "Robin",
    confidence: float = 0.95,
    track_id: int = 1,
    ts: datetime | None = None,
) -> DetectionRecord:
    """Insert a detection record and create stub image files on disk."""
    ts = ts or datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    species_dir = image_dir / species
    species_dir.mkdir(exist_ok=True)
    img_rel = f"{species}/img_{track_id}.jpg"
    thumb_rel = f"{species}/img_{track_id}_thumb.jpg"
    (image_dir / img_rel).write_bytes(b"FAKEJPEG")
    (image_dir / thumb_rel).write_bytes(b"FAKETHUMB")
    record = DetectionRecord(
        timestamp=ts,
        species=species,
        confidence=confidence,
        image_path=img_rel,
        thumbnail_path=thumb_rel,
        track_id=track_id,
        stable_frames=5,
        duration_sec=1.2,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


@pytest.fixture()
def seeded_session(session_factory, image_dir):
    """Session with several detection records pre-inserted."""
    with session_factory() as session:
        _make_record(
            session,
            image_dir,
            species="Robin",
            confidence=0.95,
            track_id=1,
            ts=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        _make_record(
            session,
            image_dir,
            species="Robin",
            confidence=0.88,
            track_id=2,
            ts=datetime(2024, 6, 2, 12, 0, 0, tzinfo=timezone.utc),
        )
        _make_record(
            session,
            image_dir,
            species="Sparrow",
            confidence=0.91,
            track_id=3,
            ts=datetime(2024, 6, 3, 12, 0, 0, tzinfo=timezone.utc),
        )
        session.commit()


@pytest.fixture()
def client(session_factory, image_dir, seeded_session):
    """TestClient with DB and image dependencies overridden."""
    from backend.main import app
    from backend.dependencies import get_session, get_image_dir

    def _override_session():
        with session_factory() as session:
            yield session

    def _override_image_dir():
        return image_dir

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_image_dir] = _override_image_dir
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


class TestListDetections:
    def test_returns_all_records(self, client):
        resp = client.get("/api/detections")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_species_filter(self, client):
        resp = client.get("/api/detections?species=Robin")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(d["species"] == "Robin" for d in data)

    def test_from_filter(self, client):
        resp = client.get("/api/detections?from=2024-06-02T00:00:00")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_to_filter(self, client):
        resp = client.get("/api/detections?to=2024-06-01T23:59:59")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_pagination(self, client):
        resp = client.get("/api/detections?limit=2&offset=0")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_results_ordered_by_timestamp_desc(self, client):
        resp = client.get("/api/detections")
        timestamps = [d["timestamp"] for d in resp.json()]
        assert timestamps == sorted(timestamps, reverse=True)


class TestGetDetection:
    def test_returns_correct_record(self, client):
        all_resp = client.get("/api/detections")
        first_id = all_resp.json()[0]["id"]
        resp = client.get(f"/api/detections/{first_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == first_id

    def test_not_found(self, client):
        resp = client.get("/api/detections/99999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class TestImages:
    def _first_id(self, client):
        return client.get("/api/detections").json()[-1]["id"]

    def test_thumbnail_returns_jpeg(self, client):
        det_id = self._first_id(client)
        resp = client.get(f"/api/images/{det_id}/thumbnail")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == b"FAKETHUMB"

    def test_full_image_returns_jpeg(self, client):
        det_id = self._first_id(client)
        resp = client.get(f"/api/images/{det_id}/full")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == b"FAKEJPEG"

    def test_thumbnail_not_found(self, client):
        resp = client.get("/api/images/99999/thumbnail")
        assert resp.status_code == 404

    def test_full_image_not_found(self, client):
        resp = client.get("/api/images/99999/full")
        assert resp.status_code == 404


class TestDownload:
    def test_download_returns_valid_zip(self, client):
        all_data = client.get("/api/detections").json()
        ids = ",".join(str(d["id"]) for d in all_data)
        resp = client.get(f"/api/images/download?ids={ids}")
        assert resp.status_code == 200
        assert "zip" in resp.headers["content-type"]
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = zf.namelist()
        assert len(names) == 3
        assert all(n.endswith(".jpg") for n in names)

    def test_download_skips_missing_ids(self, client):
        all_data = client.get("/api/detections").json()
        first_id = all_data[0]["id"]
        resp = client.get(f"/api/images/download?ids={first_id},99999")
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        assert len(zf.namelist()) == 1

    def test_download_invalid_ids(self, client):
        resp = client.get("/api/images/download?ids=abc,def")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------


class TestSystem:
    def test_returns_expected_fields(self, client):
        resp = client.get("/api/system")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_percent" in data
        assert "uptime_seconds" in data
        assert "cpu_temp_celsius" in data

    def test_cpu_percent_in_range(self, client):
        resp = client.get("/api/system")
        cpu = resp.json()["cpu_percent"]
        assert 0.0 <= cpu <= 100.0

    def test_uptime_positive(self, client):
        resp = client.get("/api/system")
        assert resp.json()["uptime_seconds"] > 0


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------


class TestSpecies:
    def test_lists_all_species(self, client):
        resp = client.get("/api/species")
        assert resp.status_code == 200
        data = resp.json()
        names = {s["species"] for s in data}
        assert names == {"Robin", "Sparrow"}

    def test_counts_correct(self, client):
        resp = client.get("/api/species")
        data = {s["species"]: s["count"] for s in resp.json()}
        assert data["Robin"] == 2
        assert data["Sparrow"] == 1

    def test_ordered_by_count_desc(self, client):
        resp = client.get("/api/species")
        counts = [s["count"] for s in resp.json()]
        assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# Camera snapshot proxy
# ---------------------------------------------------------------------------


class _FakeHttpxResponse:
    """Minimal httpx.Response stand-in for the camera proxy."""

    def __init__(
        self,
        content: bytes = b"",
        content_type: str = "image/jpeg",
        status_code: int = 200,
        json_body=None,
    ) -> None:
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status_code
        self.text = content.decode("utf-8", "replace")
        self._json_body = json_body

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json_body


class TestCamera:
    def test_snapshot_proxies_detector_jpeg(self, client, monkeypatch):
        from backend.routers import camera

        captured = {}

        def _fake_get(url, timeout):
            captured["url"] = url
            return _FakeHttpxResponse(b"JPEGBYTES")

        monkeypatch.setattr(camera.httpx, "get", _fake_get)
        resp = client.get("/api/camera/snapshot")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == b"JPEGBYTES"
        assert captured["url"].endswith("/capture")

    def test_snapshot_returns_503_when_detector_unreachable(self, client, monkeypatch):
        import httpx

        from backend.routers import camera

        def _fake_get(url, timeout):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(camera.httpx, "get", _fake_get)
        resp = client.get("/api/camera/snapshot")
        assert resp.status_code == 503

    def test_full_snapshot_proxies_detector_jpeg(self, client, monkeypatch):
        from backend.routers import camera

        captured = {}

        def _fake_get(url, timeout):
            captured["url"] = url
            return _FakeHttpxResponse(b"FULLJPEG")

        monkeypatch.setattr(camera.httpx, "get", _fake_get)
        resp = client.get("/api/camera/snapshot/full")
        assert resp.status_code == 200
        assert resp.content == b"FULLJPEG"
        assert captured["url"].endswith("/capture/full")

    def test_get_crop_proxies_detector_json(self, client, monkeypatch):
        from backend.routers import camera

        state = {"x": 1, "y": 2, "w": 900, "h": 900, "sensor_w": 4056}

        def _fake_get(url, timeout):
            return _FakeHttpxResponse(content_type="application/json", json_body=state)

        monkeypatch.setattr(camera.httpx, "get", _fake_get)
        resp = client.get("/api/camera/crop")
        assert resp.status_code == 200
        assert resp.json() == state

    def test_set_crop_proxies_post_body(self, client, monkeypatch):
        from backend.routers import camera

        captured = {}
        state = {"x": 1, "y": 2, "w": 900, "h": 900}

        def _fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return _FakeHttpxResponse(
                content_type="application/json", json_body=state
            )

        monkeypatch.setattr(camera.httpx, "post", _fake_post)
        resp = client.post(
            "/api/camera/crop", json={"nx": 0.1, "ny": 0.2, "nw": 0.3, "nh": 0.4}
        )
        assert resp.status_code == 200
        assert resp.json() == state
        assert captured["url"].endswith("/crop")
        assert captured["json"] == {"nx": 0.1, "ny": 0.2, "nw": 0.3, "nh": 0.4}

    def test_set_crop_relays_detector_400(self, client, monkeypatch):
        from backend.routers import camera

        def _fake_post(url, json, timeout):
            return _FakeHttpxResponse(b"bad body", status_code=400)

        monkeypatch.setattr(camera.httpx, "post", _fake_post)
        resp = client.post("/api/camera/crop", json={"nx": 0.1})
        assert resp.status_code == 400
