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
