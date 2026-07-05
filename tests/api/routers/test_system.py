"""Integration tests for ``GET /api/system`` (host CPU/mem/disk/temp/uptime)."""


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
