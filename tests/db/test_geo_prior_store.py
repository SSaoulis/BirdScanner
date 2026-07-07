"""Tests for the geomodel-prior persistence layer (db/geo_prior_store.py).

Uses the shared in-memory ``session_factory`` fixture (from the top-level
conftest), whose engine has already run ``init_db`` — which now also creates the
``geo_priors`` / ``geo_prior_meta`` tables.
"""

from datetime import datetime

from sqlmodel import select

from birdscanner.db.geo_prior_store import (
    has_geo_priors,
    load_geo_priors,
    location_matches,
    read_meta,
    replace_geo_priors,
)
from birdscanner.db.models import GeoPrior, GeoPriorMeta


def _count_priors(session_factory) -> int:
    """Return the number of GeoPrior rows currently stored."""
    with session_factory() as session:
        return len(session.exec(select(GeoPrior)).all())


def test_read_meta_is_none_before_any_build(session_factory):
    """No metadata row exists until the priors are first built."""
    assert read_meta(session_factory) is None
    assert has_geo_priors(session_factory) is False


def test_replace_geo_priors_writes_rows_and_meta(session_factory):
    """Every (species, week) pair is written and the meta row records the location."""
    priors = {"Robin": [0.1, 0.2, 0.3], "Blackbird": [0.4, 0.5, 0.6]}

    written = replace_geo_priors(session_factory, 51.5, -0.12, priors)

    assert written == 6
    assert _count_priors(session_factory) == 6
    assert has_geo_priors(session_factory) is True

    with session_factory() as session:
        robin = session.exec(select(GeoPrior).where(GeoPrior.species == "Robin")).all()
    by_week = sorted((r.week, r.probability) for r in robin)
    assert by_week == [(1, 0.1), (2, 0.2), (3, 0.3)]

    meta = read_meta(session_factory)
    assert meta is not None
    assert (meta.latitude, meta.longitude) == (51.5, -0.12)
    assert meta.generated_at is not None


def test_replace_geo_priors_replaces_previous_rows(session_factory):
    """A second build clears the old rows and updates the single meta row."""
    replace_geo_priors(session_factory, 51.5, -0.12, {"Robin": [0.1, 0.2]})

    replace_geo_priors(session_factory, 40.0, -3.7, {"Sparrow": [0.7, 0.8, 0.9]})

    assert _count_priors(session_factory) == 3
    with session_factory() as session:
        species = set(session.exec(select(GeoPrior.species)).all())
    assert species == {"Sparrow"}

    meta = read_meta(session_factory)
    assert meta is not None
    assert (meta.latitude, meta.longitude) == (40.0, -3.7)


def test_load_geo_priors_roundtrips_by_species_and_week(session_factory):
    """load_geo_priors returns {species: [weekly probs in week order]} — inverse of replace."""
    priors = {"Robin": [0.1, 0.2, 0.3], "Blackbird": [0.4, 0.5, 0.6]}
    replace_geo_priors(session_factory, 51.5, -0.12, priors)

    assert load_geo_priors(session_factory) == priors


def test_load_geo_priors_empty_when_nothing_stored(session_factory):
    """No stored priors yields an empty mapping (geomodel correction stays disabled)."""
    assert load_geo_priors(session_factory) == {}


def test_location_matches_within_tolerance():
    """Coordinates within the tolerance compare equal; larger deltas do not."""
    meta = GeoPriorMeta(
        id=1, latitude=51.5, longitude=-0.12, generated_at=datetime.now()
    )
    assert location_matches(meta, 51.5, -0.12) is True
    assert location_matches(meta, 51.50000001, -0.12000001) is True
    assert location_matches(meta, 51.6, -0.12) is False
    assert location_matches(None, 51.5, -0.12) is False
