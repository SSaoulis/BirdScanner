"""Integration tests for ``GET/DELETE /api/detections``.

Uses the shared ``client`` / ``make_client`` / ``detection_factory`` fixtures from
``tests/api/conftest.py`` + the top-level conftest, so no real DB or filesystem is
needed.
"""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

import pytest


class TestListDetections:
    def test_returns_all_records(self, client):
        resp = client.get("/api/detections")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_includes_both_confidences(self, client):
        resp = client.get("/api/detections")
        assert resp.status_code == 200
        record = resp.json()[0]
        assert "confidence" in record
        assert record["detection_confidence"] == pytest.approx(0.8)

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

    def test_min_confidence_filter(self, client):
        resp = client.get("/api/detections?min_confidence=0.9")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(d["confidence"] >= 0.9 for d in data)

    def test_min_confidence_out_of_range_rejected(self, client):
        resp = client.get("/api/detections?min_confidence=1.5")
        assert resp.status_code == 422

    def test_pagination(self, client):
        resp = client.get("/api/detections?limit=2&offset=0")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_results_ordered_by_timestamp_desc(self, client):
        resp = client.get("/api/detections")
        timestamps = [d["timestamp"] for d in resp.json()]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_tied_timestamps_paginate_without_duplicates(
        self, make_client, detection_factory
    ):
        """Pages must not overlap when many rows share one timestamp.

        Without a deterministic tiebreaker SQLite can return tied rows in a
        different order per query, so offset-based pages overlap and the UI
        shows the same detection twice. The ``id`` tiebreaker keeps the order
        stable so every paginated page is disjoint.
        """
        same_ts = datetime(2024, 7, 1, 9, 0, 0, tzinfo=timezone.utc)
        for track_id in range(10):
            detection_factory(track_id=track_id, ts=same_ts)

        client: TestClient = make_client()
        seen_ids: list[int] = []
        for offset in range(0, 10, 3):
            page = client.get(f"/api/detections?limit=3&offset={offset}").json()
            seen_ids.extend(d["id"] for d in page)

        assert len(seen_ids) == len(set(seen_ids)) == 10


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


class TestDeleteDetection:
    def test_proxies_delete_to_detector(self, client, monkeypatch, fake_httpx_response):
        from birdscanner.api.routers import detections

        captured = {}

        def _fake_delete(url, timeout):
            captured["url"] = url
            return fake_httpx_response(status_code=204)

        monkeypatch.setattr(detections.httpx, "delete", _fake_delete)
        resp = client.delete("/api/detections/123")
        assert resp.status_code == 204
        assert captured["url"].endswith("/detections/123")

    def test_relays_404_from_detector(self, client, monkeypatch, fake_httpx_response):
        from birdscanner.api.routers import detections

        monkeypatch.setattr(
            detections.httpx,
            "delete",
            lambda url, timeout: fake_httpx_response(status_code=404),
        )
        resp = client.delete("/api/detections/123")
        assert resp.status_code == 404

    def test_returns_503_when_detector_unreachable(self, client, monkeypatch):
        import httpx

        from birdscanner.api.routers import detections

        def _fake_delete(url, timeout):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(detections.httpx, "delete", _fake_delete)
        resp = client.delete("/api/detections/123")
        assert resp.status_code == 503
