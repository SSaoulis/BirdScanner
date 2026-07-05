"""Unit tests for the geomodel <-> classifier label crosswalk (pure functions)."""

from birdscanner.ml.geomodel import build_name_mapping, normalize_common_name


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
