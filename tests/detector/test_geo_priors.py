"""Tests for the startup geo-prior orchestrator (detector/geo_priors.py).

Uses the shared in-memory ``session_factory`` fixture and an injected fake
``compute`` so no ONNX model / onnxruntime is needed.
"""

from birdscanner.db.geo_prior_store import read_meta
from birdscanner.detector.geo_priors import refresh_geo_priors


def _fake_compute(priors):
    """Return a compute callable that records its calls and yields ``priors``."""
    calls: list[tuple[float, float]] = []

    def _compute(lat, lon):
        calls.append((lat, lon))
        return priors

    return _compute, calls


def test_refresh_skips_when_no_location(session_factory):
    """With no location set, nothing is computed or stored."""
    compute, calls = _fake_compute({"Robin": [0.1, 0.2]})

    built = refresh_geo_priors(session_factory, None, None, compute=compute)

    assert built is False
    assert calls == []
    assert read_meta(session_factory) is None


def test_refresh_builds_when_empty(session_factory):
    """A first run with a location computes and stores the prior."""
    compute, calls = _fake_compute({"Robin": [0.1, 0.2], "Wren": [0.3, 0.4]})

    built = refresh_geo_priors(session_factory, 51.5, -0.12, compute=compute)

    assert built is True
    assert calls == [(51.5, -0.12)]
    meta = read_meta(session_factory)
    assert meta is not None
    assert (meta.latitude, meta.longitude) == (51.5, -0.12)


def test_refresh_skips_when_location_matches(session_factory):
    """A second run at the same location is a no-op (no recompute)."""
    compute, calls = _fake_compute({"Robin": [0.1, 0.2]})
    refresh_geo_priors(session_factory, 51.5, -0.12, compute=compute)

    built = refresh_geo_priors(session_factory, 51.5, -0.12, compute=compute)

    assert built is False
    assert calls == [(51.5, -0.12)]  # only the first build computed


def test_refresh_rebuilds_when_location_changes(session_factory):
    """Changing the configured location recomputes and replaces the prior."""
    compute, calls = _fake_compute({"Robin": [0.1, 0.2]})
    refresh_geo_priors(session_factory, 51.5, -0.12, compute=compute)

    built = refresh_geo_priors(session_factory, 40.0, -3.7, compute=compute)

    assert built is True
    assert calls == [(51.5, -0.12), (40.0, -3.7)]
    meta = read_meta(session_factory)
    assert meta is not None
    assert (meta.latitude, meta.longitude) == (40.0, -3.7)


def test_refresh_swallows_compute_errors(session_factory):
    """A computation failure (e.g. missing model) never crashes startup."""

    def _boom(lat, lon):
        raise FileNotFoundError("geomodel .onnx missing")

    built = refresh_geo_priors(session_factory, 51.5, -0.12, compute=_boom)

    assert built is False
    assert read_meta(session_factory) is None


def test_refresh_stores_nothing_for_empty_priors(session_factory):
    """A compute that returns no species stores nothing and reports skipped."""
    compute, _ = _fake_compute({})

    built = refresh_geo_priors(session_factory, 51.5, -0.12, compute=compute)

    assert built is False
    assert read_meta(session_factory) is None
