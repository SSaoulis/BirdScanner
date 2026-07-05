"""Integration tests for ``GET /api/system`` (host CPU/mem/disk/temp/uptime)."""

from collections import namedtuple

import pytest

from birdscanner.api.routers import system

# Mirrors psutil's ``shwtemp`` named tuple (only ``.current`` is read).
_Temp = namedtuple("_Temp", ["label", "current", "high", "critical"])


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


class TestReadCpuTemp:
    """Unit tests for ``_read_cpu_temp`` covering each sensor-lookup branch."""

    def test_returns_none_when_sensors_unavailable(self, monkeypatch):
        """Platforms without ``sensors_temperatures`` (AttributeError) yield None."""

        def _raise_attribute_error():
            raise AttributeError("not available on this platform")

        monkeypatch.setattr(
            system.psutil, "sensors_temperatures", _raise_attribute_error, raising=False
        )
        assert system._read_cpu_temp() is None

    def test_returns_none_when_no_sensors_reported(self, monkeypatch):
        """An empty sensor mapping yields None."""
        monkeypatch.setattr(
            system.psutil, "sensors_temperatures", lambda: {}, raising=False
        )
        assert system._read_cpu_temp() is None

    def test_prefers_known_cpu_sensor_keys(self, monkeypatch):
        """A preferred key (e.g. ``coretemp``) is used ahead of other sensors."""
        temps = {
            "acpitz": [_Temp("", 40.0, None, None)],
            "coretemp": [_Temp("Core 0", 55.5, 90.0, 100.0)],
        }
        monkeypatch.setattr(
            system.psutil, "sensors_temperatures", lambda: temps, raising=False
        )
        assert system._read_cpu_temp() == pytest.approx(55.5)

    def test_falls_back_to_first_available_sensor(self, monkeypatch):
        """With no preferred key present, the first available sensor is used."""
        temps = {"nvme": [_Temp("Composite", 47.0, None, None)]}
        monkeypatch.setattr(
            system.psutil, "sensors_temperatures", lambda: temps, raising=False
        )
        assert system._read_cpu_temp() == pytest.approx(47.0)

    def test_returns_none_when_all_sensor_lists_empty(self, monkeypatch):
        """Keys present but with empty entry lists yield None (nothing to read)."""
        monkeypatch.setattr(
            system.psutil,
            "sensors_temperatures",
            lambda: {"cpu_thermal": []},
            raising=False,
        )
        assert system._read_cpu_temp() is None
