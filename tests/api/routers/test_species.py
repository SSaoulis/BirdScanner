"""Integration tests for ``GET /api/species`` (species list with counts)."""


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
