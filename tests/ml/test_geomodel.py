"""Unit tests for the geomodel <-> classifier label crosswalk (pure functions)."""

import json
from datetime import datetime

import numpy as np
import pytest

from birdscanner.ml.geomodel import (
    NUM_WEEKS,
    GeoPriorAdjuster,
    build_name_mapping,
    build_prior_matrix,
    geomodel_posterior,
    load_name_mapping,
    normalize_common_name,
    project_to_classifier,
    week_of_year,
)


def test_normalize_folds_case_accents_and_punctuation():
    """Case, diacritics, apostrophes/hyphens/spaces all collapse to one key."""
    assert normalize_common_name("Audouin's Gull") == normalize_common_name(
        "Audouins gull"
    )
    assert normalize_common_name("Arabian Green Bee-eater") == "arabiangreenbeeeater"
    assert normalize_common_name("Rüppell's Vulture") == "ruppellsvulture"


def test_normalize_unifies_british_grey_to_gray():
    """British 'grey' and American 'gray' spellings compare equal."""
    assert normalize_common_name("Grey Heron") == normalize_common_name("Gray Heron")
    assert normalize_common_name("Greylag Goose") == "graylaggoose"


def _geo(common):
    """Build a minimal geomodel label row for the given common name."""
    return {"id": "1", "scientific": "Genus species", "common": common}


def test_build_name_mapping_auto_matches_and_reports_unmatched():
    """Normalisation-equal names map (keyed by classifier label); the rest are unmatched."""
    geomodel = [_geo("Grey Heron"), _geo("Eurasian Blackbird")]
    classifier = ["Grey heron", "Common blackbird"]

    mapping, unmatched = build_name_mapping(geomodel, classifier)

    # Keyed by the classifier label -> the geomodel's own common name.
    assert mapping == {"Grey heron": "Grey Heron"}
    # 'Common blackbird' is a genuine synonym normalisation can't bridge.
    assert unmatched == ["Common blackbird"]


def test_build_name_mapping_applies_and_preserves_overrides():
    """Curated overrides add the synonym and drop it from the unmatched list."""
    geomodel = [_geo("Grey Heron"), _geo("Eurasian Blackbird")]
    classifier = ["Grey heron", "Common blackbird"]
    overrides = {"Common blackbird": "Eurasian Blackbird"}

    mapping, unmatched = build_name_mapping(geomodel, classifier, overrides)

    assert mapping == {
        "Grey heron": "Grey Heron",
        "Common blackbird": "Eurasian Blackbird",
    }
    assert unmatched == []


def test_build_name_mapping_allows_many_classifier_classes_per_geomodel_species():
    """Several classifier classes may map to one geomodel species (e.g. an eBird lump)."""
    geomodel = [_geo("Redpoll")]
    classifier = ["Common redpoll", "Arctic redpoll"]
    overrides = {"Common redpoll": "Redpoll", "Arctic redpoll": "Redpoll"}

    mapping, unmatched = build_name_mapping(geomodel, classifier, overrides)

    # Both classifier classes coexist pointing at the same geomodel species —
    # impossible if the map were keyed by geomodel name.
    assert mapping == {"Common redpoll": "Redpoll", "Arctic redpoll": "Redpoll"}
    assert unmatched == []


def test_build_name_mapping_first_geomodel_row_wins_on_collision():
    """When two geomodel rows normalise alike, the first (by order) is kept."""
    geomodel = [_geo("Grey Heron"), _geo("Gray heron")]
    classifier = ["Grey heron"]

    mapping, unmatched = build_name_mapping(geomodel, classifier)

    assert mapping == {"Grey heron": "Grey Heron"}
    assert unmatched == []


def test_load_name_mapping_reads_json(tmp_path):
    """load_name_mapping returns the {classifier_label: geomodel_common_name} object."""
    path = tmp_path / "map.json"
    path.write_text(
        json.dumps({"Grey heron": "Grey Heron", "Robin": "European Robin"}),
        encoding="utf-8",
    )

    assert load_name_mapping(str(path)) == {
        "Grey heron": "Grey Heron",
        "Robin": "European Robin",
    }


def test_project_to_classifier_selects_the_mapped_column():
    """Each classifier label borrows its mapped geomodel species' 48-week column."""
    geomodel = [_geo("Grey Heron"), _geo("European Robin")]
    # (weeks, species): column 0 = heron, column 1 = robin.
    predictions = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=np.float32)
    mapping = {"Grey heron": "Grey Heron", "Robin": "European Robin"}

    priors = project_to_classifier(predictions, geomodel, mapping)

    assert priors["Grey heron"] == pytest.approx([0.1, 0.2, 0.3])
    assert priors["Robin"] == pytest.approx([0.9, 0.8, 0.7])


def test_project_to_classifier_shares_a_column_many_to_one():
    """Several classifier classes may map to (and share) one geomodel column."""
    geomodel = [_geo("Redpoll")]
    predictions = np.array([[0.4], [0.5]], dtype=np.float32)
    mapping = {"Common redpoll": "Redpoll", "Arctic redpoll": "Redpoll"}

    priors = project_to_classifier(predictions, geomodel, mapping)

    assert set(priors) == {"Common redpoll", "Arctic redpoll"}
    assert priors["Common redpoll"] == pytest.approx([0.4, 0.5])
    assert priors["Arctic redpoll"] == pytest.approx([0.4, 0.5])


def test_project_to_classifier_drops_labels_absent_from_geomodel():
    """A classifier label whose mapped geomodel species is missing is dropped."""
    geomodel = [_geo("Grey Heron")]
    predictions = np.array([[0.1], [0.2]], dtype=np.float32)
    mapping = {"Grey heron": "Grey Heron", "Robin": "European Robin"}

    priors = project_to_classifier(predictions, geomodel, mapping)

    assert set(priors) == {"Grey heron"}


def test_project_to_classifier_first_geomodel_row_wins_on_duplicate_common_name():
    """On a duplicate geomodel common name, the first row's column is used."""
    geomodel = [_geo("American Barn Owl"), _geo("American Barn Owl")]
    predictions = np.array([[0.6, 0.1]], dtype=np.float32)
    mapping = {"Barn owl": "American Barn Owl"}

    priors = project_to_classifier(predictions, geomodel, mapping)

    assert set(priors) == {"Barn owl"}
    assert priors["Barn owl"] == pytest.approx([0.6])


# ---------------------------------------------------------------------------
# Runtime Bayesian update: week_of_year / build_prior_matrix /
# geomodel_posterior / GeoPriorAdjuster
# ---------------------------------------------------------------------------


def test_week_of_year_bins_into_month_quarters():
    """Each month splits into 4 weeks; the index stays within 1..NUM_WEEKS."""
    assert week_of_year(datetime(2026, 1, 1)) == 1  # first quarter of January
    assert week_of_year(datetime(2026, 1, 8)) == 2  # second quarter
    assert week_of_year(datetime(2026, 1, 31)) == 4  # capped at the 4th quarter
    assert week_of_year(datetime(2026, 2, 1)) == 5  # rolls into February
    assert week_of_year(datetime(2026, 12, 31)) == NUM_WEEKS  # last week of the year


def _weekly(value: float) -> list[float]:
    """A constant 48-week prior vector (the store always writes NUM_WEEKS weeks)."""
    return [value] * NUM_WEEKS


def test_build_prior_matrix_aligns_floors_and_fills_unmapped():
    """Mapped species land in their class column (floored); unmapped get the neutral prior."""
    idx_to_class = {0: "Robin", 1: "Blackbird", 2: "Unknown"}
    priors = {"Robin": _weekly(0.6), "Blackbird": _weekly(0.0)}

    matrix = build_prior_matrix(priors, idx_to_class, floor=1e-3, unmapped_prior=1.0)

    assert matrix.shape == (NUM_WEEKS, 3)
    assert matrix[:, 0] == pytest.approx([0.6] * NUM_WEEKS)  # mapped
    assert matrix[:, 1] == pytest.approx([1e-3] * NUM_WEEKS)  # floored up from 0.0
    assert matrix[:, 2] == pytest.approx([1.0] * NUM_WEEKS)  # unmapped -> neutral


def test_build_prior_matrix_ignores_wrong_length_vectors():
    """A weekly vector that is not NUM_WEEKS long is skipped (left at the neutral prior)."""
    idx_to_class = {0: "Robin"}
    matrix = build_prior_matrix({"Robin": [0.5, 0.5]}, idx_to_class, unmapped_prior=1.0)
    assert matrix[:, 0] == pytest.approx([1.0] * NUM_WEEKS)


def test_geomodel_posterior_matches_the_formula():
    """Posterior = normalized product; unnormalized is the raw product."""
    probs = np.array([0.6, 0.4])
    prior = np.array([0.1, 0.9])

    posterior, unnormalized = geomodel_posterior(probs, prior)

    assert unnormalized == pytest.approx([0.06, 0.36])
    assert posterior == pytest.approx([0.06 / 0.42, 0.36 / 0.42])
    assert float(posterior.sum()) == pytest.approx(1.0)


def test_geomodel_posterior_zero_product_falls_back_to_classifier():
    """When the product sums to 0 the classifier probabilities are returned unchanged."""
    probs = np.array([0.7, 0.3])
    prior = np.array([0.0, 0.0])

    posterior, unnormalized = geomodel_posterior(probs, prior)

    assert posterior == pytest.approx([0.7, 0.3])
    assert unnormalized == pytest.approx([0.0, 0.0])


def test_geo_prior_adjuster_flips_prediction_towards_the_likely_species():
    """A strong prior for a runner-up class flips the prediction and records both picks."""
    idx_to_class = {0: "Vagrant", 1: "Local robin"}
    # Classifier narrowly prefers the out-of-range Vagrant; the geo prior for it is
    # tiny while the Local robin is common, so the posterior should flip.
    priors = {"Vagrant": _weekly(0.001), "Local robin": _weekly(0.9)}
    adjuster = GeoPriorAdjuster(priors, idx_to_class, floor=1e-4, top_k=2)

    result = adjuster.adjust(np.array([0.55, 0.45]), week=1)

    assert result.classifier_species == "Vagrant"  # classifier's own pick
    assert result.classifier_confidence == pytest.approx(0.55)
    assert result.species == "Local robin"  # geomodel-corrected pick
    assert result.confidence > 0.5
    # top_scores are the pre-normalised product, descending.
    assert [name for name, _ in result.top_scores] == ["Local robin", "Vagrant"]
    assert result.top_scores[0][1] == pytest.approx(0.45 * 0.9)


def test_geo_prior_adjuster_clamps_out_of_range_week():
    """A week beyond NUM_WEEKS is clamped rather than indexing past the matrix."""
    adjuster = GeoPriorAdjuster({"Robin": _weekly(0.5)}, {0: "Robin"})
    # Must not raise; returns a valid prediction for the last week's row.
    result = adjuster.adjust(np.array([1.0]), week=NUM_WEEKS + 5)
    assert result.species == "Robin"
