"""Unit tests for the geomodel <-> classifier label crosswalk (pure functions)."""

import json

import numpy as np
import pytest

from birdscanner.ml.geomodel import (
    build_name_mapping,
    load_name_mapping,
    normalize_common_name,
    project_to_classifier,
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
