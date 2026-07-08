"""Integration tests for ``GET /api/species`` (species list with counts)."""

from datetime import datetime

from birdscanner.api.routers.species import _current_geo_week
from birdscanner.db.models import GeoPrior, GeoPriorMeta


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


class TestVocabulary:
    def test_proxies_vocabulary_from_detector(
        self, client, monkeypatch, fake_httpx_response
    ):
        from birdscanner.api.routers import species

        monkeypatch.setattr(
            species.httpx,
            "get",
            lambda url, timeout: fake_httpx_response(
                status_code=200, json_body={"species": ["Robin", "Sparrow", "Wren"]}
            ),
        )
        resp = client.get("/api/species/vocabulary")
        assert resp.status_code == 200
        assert resp.json() == ["Robin", "Sparrow", "Wren"]

    def test_returns_503_when_detector_unreachable(self, client, monkeypatch):
        import httpx

        from birdscanner.api.routers import species

        def _fake_get(url, timeout):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(species.httpx, "get", _fake_get)
        resp = client.get("/api/species/vocabulary")
        assert resp.status_code == 503


def _seed_priors(session_factory, rows):
    """Insert ``(species, week, probability)`` GeoPrior rows into the test DB."""
    with session_factory() as session:
        for name, week, prob in rows:
            session.add(GeoPrior(species=name, week=week, probability=prob))
        session.commit()


class TestExpected:
    def test_empty_and_no_location_before_priors_built(self, make_client):
        client = make_client()
        resp = client.get("/api/species/expected")
        assert resp.status_code == 200
        body = resp.json()
        assert body["species"] == []
        assert body["latitude"] is None
        assert body["longitude"] is None
        assert 1 <= body["week"] <= 48

    def test_returns_current_week_ordered_by_score_desc(
        self, session_factory, make_client
    ):
        week = _current_geo_week(datetime.now())
        other_week = 1 if week != 1 else 2
        _seed_priors(
            session_factory,
            [
                ("Robin", week, 0.9),
                ("Sparrow", week, 0.5),
                ("Wren", week, 0.7),
                # A different week must not leak into the current-week ranking.
                ("Owl", other_week, 0.99),
            ],
        )
        client = make_client()
        body = client.get("/api/species/expected").json()
        assert body["week"] == week
        assert [s["species"] for s in body["species"]] == ["Robin", "Wren", "Sparrow"]

    def test_honours_limit(self, session_factory, make_client):
        week = _current_geo_week(datetime.now())
        _seed_priors(session_factory, [(f"S{i}", week, i / 10) for i in range(1, 6)])
        client = make_client()
        body = client.get("/api/species/expected", params={"limit": 2}).json()
        assert [s["species"] for s in body["species"]] == ["S5", "S4"]

    def test_reports_configured_location(self, session_factory, make_client):
        week = _current_geo_week(datetime.now())
        with session_factory() as session:
            session.add(
                GeoPriorMeta(
                    id=1, latitude=51.5, longitude=-0.12, generated_at=datetime.now()
                )
            )
            session.commit()
        _seed_priors(session_factory, [("Robin", week, 0.8)])
        client = make_client()
        body = client.get("/api/species/expected").json()
        assert body["latitude"] == 51.5
        assert body["longitude"] == -0.12
        assert [s["species"] for s in body["species"]] == ["Robin"]
