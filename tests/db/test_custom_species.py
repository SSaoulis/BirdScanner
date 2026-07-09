"""Tests for ``db.custom_species`` (the user-added species-label store).

Uses the shared in-memory ``session_factory`` fixture (top-level conftest), so no
real detector or filesystem is needed.
"""

import pytest

from birdscanner.db.custom_species import add_custom_species, list_custom_species


def test_list_is_empty_initially(session_factory):
    """No custom species are stored on a fresh database."""
    assert list_custom_species(session_factory) == []


def test_add_then_list(session_factory):
    """An added label is returned canonical and appears in the listing."""
    stored = add_custom_species(session_factory, "Hoopoe")
    assert stored == "Hoopoe"
    assert list_custom_species(session_factory) == ["Hoopoe"]


def test_add_trims_whitespace(session_factory):
    """Leading/trailing whitespace is stripped before storing."""
    assert add_custom_species(session_factory, "  Wryneck  ") == "Wryneck"
    assert list_custom_species(session_factory) == ["Wryneck"]


def test_add_is_idempotent_exact(session_factory):
    """Re-adding the same label writes no second row."""
    add_custom_species(session_factory, "Hoopoe")
    assert add_custom_species(session_factory, "Hoopoe") == "Hoopoe"
    assert list_custom_species(session_factory) == ["Hoopoe"]


def test_add_dedupes_case_insensitively_keeping_original(session_factory):
    """A case-variant reuses the first-stored canonical label, adding no row."""
    add_custom_species(session_factory, "Hoopoe")
    result = add_custom_species(session_factory, "hoopoe")
    assert result == "Hoopoe"  # original casing preserved
    assert list_custom_species(session_factory) == ["Hoopoe"]


def test_add_empty_raises(session_factory):
    """An empty / whitespace-only label is rejected."""
    with pytest.raises(ValueError):
        add_custom_species(session_factory, "   ")
