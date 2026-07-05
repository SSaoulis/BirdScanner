"""Tests for the geolocation model stub and the pure prior-generation helpers."""

import json

import numpy as np
import pytest

from birdscanner.ml.geo_priors import (
    WEEKS_IN_YEAR,
    current_week,
    generate_all_weeks,
    species_signature,
)
from birdscanner.ml.geolocation import (
    PlaceholderGeolocationModel,
    load_species_order,
)

_SPECIES = ["Robin", "Jay", "Unknown"]


def test_placeholder_returns_uniform_normalized_vector():
    model = PlaceholderGeolocationModel(_SPECIES)
    vec = model.predict_week(51.5, -0.12, 7)
    assert vec.shape == (len(_SPECIES),)
    assert vec.dtype == np.float32
    assert np.isclose(vec.sum(), 1.0)
    assert np.allclose(vec, vec[0])  # uniform


def test_placeholder_species_order_is_a_copy():
    model = PlaceholderGeolocationModel(_SPECIES)
    order = model.species_order
    order.append("mutated")
    assert model.species_order == _SPECIES  # internal state untouched


def test_placeholder_rejects_empty_species_order():
    with pytest.raises(ValueError, match="non-empty"):
        PlaceholderGeolocationModel([])


def test_generate_all_weeks_covers_the_year_and_reports_progress():
    model = PlaceholderGeolocationModel(_SPECIES)
    seen: list[int] = []
    vectors = generate_all_weeks(model, 51.5, -0.12, on_progress=seen.append)

    assert set(vectors) == set(range(1, WEEKS_IN_YEAR + 1))
    assert seen == list(range(1, WEEKS_IN_YEAR + 1))
    assert all(v.shape == (len(_SPECIES),) for v in vectors.values())


def test_generate_all_weeks_without_progress_callback():
    model = PlaceholderGeolocationModel(_SPECIES)
    vectors = generate_all_weeks(model, 0.0, 0.0)
    assert len(vectors) == WEEKS_IN_YEAR


def test_species_signature_is_stable_and_order_sensitive():
    a = species_signature(["Robin", "Jay"])
    assert a == species_signature(["Robin", "Jay"])
    assert a != species_signature(["Jay", "Robin"])
    assert a != species_signature(["Robin", "Jay", "Wren"])


def test_current_week_is_clamped_to_52():
    week = current_week()
    assert 1 <= week <= WEEKS_IN_YEAR


def test_load_species_order_sorts_by_index(tmp_path):
    path = tmp_path / "class_to_idx.json"
    path.write_text(json.dumps({"Jay": 1, "Robin": 0, "Unknown": 2}), encoding="utf-8")
    assert load_species_order(str(path)) == ["Robin", "Jay", "Unknown"]
