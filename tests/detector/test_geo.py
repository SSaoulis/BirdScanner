"""Tests for the startup geolocation-prior bootstrap coordination."""

import numpy as np

from birdscanner.detector.geo import GeoPriorProvider, bootstrap_geo_priors
from birdscanner.detector.settings import default_settings, merge_settings


class FakeGeoModel:
    """A recording geolocation model: counts predict_week calls."""

    def __init__(self, species_order, version="v-test"):
        self._species_order = list(species_order)
        self._version = version
        self.calls = 0

    @property
    def species_order(self):
        return list(self._species_order)

    @property
    def model_version(self):
        return self._version

    def predict_week(self, latitude, longitude, week):
        self.calls += 1
        return np.full(len(self._species_order), 0.5, dtype=np.float32)


def _settings(lat=51.5, lon=-0.12):
    return merge_settings(default_settings(), {"latitude": lat, "longitude": lon})


def test_bootstrap_generates_on_first_run(tmp_path, monkeypatch):
    monkeypatch.setenv("GEO_DB_PATH", str(tmp_path / "geo.db"))
    model = FakeGeoModel(["Robin", "Jay"])

    provider = bootstrap_geo_priors(_settings(), model)

    assert isinstance(provider, GeoPriorProvider)
    assert model.calls == 52  # all weeks generated
    assert provider.week_vector(1) is not None
    assert provider.week_vector(52) is not None


def test_bootstrap_is_cache_hit_when_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("GEO_DB_PATH", str(tmp_path / "geo.db"))

    first = FakeGeoModel(["Robin", "Jay"])
    bootstrap_geo_priors(_settings(), first)
    assert first.calls == 52

    # Same location + species: the second boot must not regenerate.
    second = FakeGeoModel(["Robin", "Jay"])
    bootstrap_geo_priors(_settings(), second)
    assert second.calls == 0


def test_bootstrap_regenerates_when_location_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("GEO_DB_PATH", str(tmp_path / "geo.db"))

    first = FakeGeoModel(["Robin", "Jay"])
    bootstrap_geo_priors(_settings(lat=10.0), first)

    moved = FakeGeoModel(["Robin", "Jay"])
    bootstrap_geo_priors(_settings(lat=20.0), moved)
    assert moved.calls == 52


def test_bootstrap_regenerates_when_species_signature_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("GEO_DB_PATH", str(tmp_path / "geo.db"))

    first = FakeGeoModel(["Robin", "Jay"])
    bootstrap_geo_priors(_settings(), first)

    # A changed class set invalidates the cache even at the same location.
    changed = FakeGeoModel(["Robin", "Jay", "Wren"])
    bootstrap_geo_priors(_settings(), changed)
    assert changed.calls == 52
