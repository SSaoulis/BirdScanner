"""Tests for the network router: history + speed-test routes, and the underlying
passive sampler / throughput math (previously untested — the route tests stub the
speed test out, so ``run_speed_test`` and ``_Sampler`` are exercised directly here).
"""

from types import SimpleNamespace

import pytest

from birdscanner.api.routers import network
from birdscanner.api.routers.network import SpeedTestResult


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class TestNetworkHistory:
    def test_returns_interval_and_samples(self, client):
        resp = client.get("/api/network/history?range=5m")
        assert resp.status_code == 200
        data = resp.json()
        assert "interval_sec" in data
        assert isinstance(data["samples"], list)

    def test_default_range_is_valid(self, client):
        resp = client.get("/api/network/history")
        assert resp.status_code == 200

    def test_rejects_unknown_range(self, client):
        resp = client.get("/api/network/history?range=bogus")
        assert resp.status_code == 400


class TestNetworkSpeedTest:
    def test_returns_measured_rates(self, client, monkeypatch):
        def _fake_run() -> SpeedTestResult:
            return SpeedTestResult(
                download_mbps=12.5,
                upload_mbps=3.25,
                download_bytes=1_048_576,
                upload_bytes=262_144,
                ran_at=1_700_000_000.0,
            )

        monkeypatch.setattr(network, "run_speed_test", _fake_run)
        resp = client.post("/api/network/speedtest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["download_mbps"] == 12.5
        assert data["upload_mbps"] == 3.25

    def test_unreachable_endpoint_returns_503(self, client, monkeypatch):
        import httpx

        def _boom() -> None:
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(network, "run_speed_test", _boom)
        resp = client.post("/api/network/speedtest")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Passive sampler
# ---------------------------------------------------------------------------


class TestSampler:
    def test_first_reading_only_seeds_baseline(self, monkeypatch):
        """The very first sample records no throughput (just the counter baseline)."""
        sampler = network._Sampler(interval=3.0, maxlen=100)
        monkeypatch.setattr(
            network.psutil,
            "net_io_counters",
            lambda: SimpleNamespace(bytes_recv=1000, bytes_sent=500),
        )
        monkeypatch.setattr(network.time, "time", lambda: 100.0)
        sampler._sample_once()
        assert sampler.history(3600) == []

    def test_derives_kbps_from_counter_delta(self, monkeypatch):
        """The second sample derives kbps from the byte-counter delta over elapsed time."""
        sampler = network._Sampler(interval=3.0, maxlen=100)
        clock = {"t": 100.0}
        counters = {"v": SimpleNamespace(bytes_recv=1000, bytes_sent=500)}
        monkeypatch.setattr(network.time, "time", lambda: clock["t"])
        monkeypatch.setattr(network.psutil, "net_io_counters", lambda: counters["v"])

        sampler._sample_once()  # seed baseline at t=100
        clock["t"] = 101.0  # one second later
        counters["v"] = SimpleNamespace(bytes_recv=1000 + 1250, bytes_sent=500 + 625)
        sampler._sample_once()

        samples = sampler.history(3600)
        assert len(samples) == 1
        # 1250 bytes * 8 bits / 1 s / 1000 = 10 kbps; 625 bytes -> 5 kbps.
        assert samples[0].rx_kbps == pytest.approx(10.0)
        assert samples[0].tx_kbps == pytest.approx(5.0)

    def test_history_filters_to_window(self, monkeypatch):
        """history() returns only samples newer than the cutoff."""
        sampler = network._Sampler(interval=3.0, maxlen=100)
        sampler._samples.append(network.NetworkSample(t=100.0, rx_kbps=1, tx_kbps=1))
        sampler._samples.append(network.NetworkSample(t=200.0, rx_kbps=2, tx_kbps=2))
        monkeypatch.setattr(network.time, "time", lambda: 250.0)

        assert len(sampler.history(100)) == 1  # cutoff 150 -> only t=200
        assert len(sampler.history(1000)) == 2


# ---------------------------------------------------------------------------
# Throughput helpers + run_speed_test
# ---------------------------------------------------------------------------


class TestSpeedTestInternals:
    def test_mbps_computes_throughput(self):
        assert network._mbps(1_000_000, 1.0) == pytest.approx(8.0)

    def test_mbps_zero_elapsed_is_zero(self):
        assert network._mbps(1000, 0.0) == 0.0

    def test_run_speed_test_measures_both_legs(self, monkeypatch):
        """run_speed_test drives httpx and reports the bytes transferred each way."""

        class _FakeResp:
            def __init__(self, content: bytes = b"") -> None:
                self.content = content

            def raise_for_status(self) -> None:
                return None

        class _FakeClient:
            def __init__(self, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url, params):
                return _FakeResp(b"\0" * params["bytes"])

            def post(self, url, content):
                return _FakeResp(b"")

        monkeypatch.setattr(network.httpx, "Client", _FakeClient)

        result = network.run_speed_test()
        assert result.download_bytes == network._DOWNLOAD_BYTES
        assert result.upload_bytes == network._UPLOAD_BYTES
        assert result.download_mbps >= 0.0
        assert result.upload_mbps >= 0.0
