"""Tests for the separate geolocation-prior store (schema + blob round-trip)."""

from datetime import datetime

import numpy as np
import pytest

from birdscanner.db.geo_store import GeoMeta, GeoPriorStore, geo_db_path


def _meta(lat: float = 51.5, lon: float = -0.12) -> GeoMeta:
    """Build a GeoMeta with a small species order for tests."""
    return GeoMeta(
        latitude=lat,
        longitude=lon,
        species_order=["Robin", "Jay", "Unknown"],
        species_signature="sig-abc",
        model_version="placeholder-1",
        generated_at=datetime(2026, 7, 5, 12, 0, 0),
    )


def _vectors(n_species: int = 3) -> dict[int, np.ndarray]:
    """52 distinct float32 vectors keyed by week."""
    return {
        week: np.full(n_species, week / 100.0, dtype=np.float32)
        for week in range(1, 53)
    }


def test_write_then_read_meta_round_trips(tmp_path):
    """read_meta returns exactly what write_priors stored."""
    store = GeoPriorStore(str(tmp_path / "geo.db"))
    meta = _meta()
    store.write_priors(meta, _vectors())

    loaded = store.read_meta()
    assert loaded is not None
    assert loaded.latitude == meta.latitude
    assert loaded.longitude == meta.longitude
    assert loaded.species_order == meta.species_order
    assert loaded.species_signature == meta.species_signature
    assert loaded.model_version == meta.model_version
    assert loaded.generated_at == meta.generated_at


def test_read_meta_empty_store_returns_none(tmp_path):
    """A fresh store has no cached metadata."""
    store = GeoPriorStore(str(tmp_path / "geo.db"))
    assert store.read_meta() is None


def test_get_week_vector_round_trips_as_float32(tmp_path):
    """Stored vectors read back bit-identical as writable float32 arrays."""
    store = GeoPriorStore(str(tmp_path / "geo.db"))
    vectors = _vectors()
    store.write_priors(_meta(), vectors)

    got = store.get_week_vector(23)
    assert got is not None
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, vectors[23])
    # frombuffer views are read-only; the store must hand back a writable copy.
    got[0] = 9.0  # must not raise


def test_get_week_vector_missing_week_returns_none(tmp_path):
    """A week outside the stored range yields None, not an error."""
    store = GeoPriorStore(str(tmp_path / "geo.db"))
    store.write_priors(_meta(), _vectors())
    assert store.get_week_vector(53) is None


def test_write_priors_replaces_previous_cache(tmp_path):
    """Re-writing for a new location clears the old rows (no stale weeks)."""
    store = GeoPriorStore(str(tmp_path / "geo.db"))
    store.write_priors(_meta(lat=10.0), _vectors())

    store.write_priors(_meta(lat=20.0), {1: np.zeros(3, dtype=np.float32)})

    loaded = store.read_meta()
    assert loaded is not None and loaded.latitude == 20.0
    # Only week 1 remains from the second write.
    assert store.get_week_vector(1) is not None
    assert store.get_week_vector(23) is None


def test_geo_db_path_env_override(monkeypatch):
    monkeypatch.setenv("GEO_DB_PATH", "/tmp/custom_geo.db")
    assert geo_db_path() == "/tmp/custom_geo.db"


def test_geo_db_path_default(monkeypatch):
    monkeypatch.delenv("GEO_DB_PATH", raising=False)
    assert geo_db_path() == "/data/geo_priors.db"


@pytest.mark.usefixtures("engine")
def test_geo_store_does_not_touch_detections_metadata(tmp_path):
    """Creating the geo store must not create the detections table in its DB."""
    from sqlalchemy import inspect

    store = GeoPriorStore(str(tmp_path / "geo.db"))
    table_names = set(inspect(store._engine).get_table_names())  # noqa: SLF001
    assert table_names == {"geo_meta", "geo_week_prior"}
    assert "detections" not in table_names
