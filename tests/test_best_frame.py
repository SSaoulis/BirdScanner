"""Unit tests for BestFrameSelector."""

import numpy as np

from birdscanner.ml.best_frame import BestFrameSelector


def _frame(value: int) -> np.ndarray:
    """A tiny solid-colour frame tagged by ``value`` for identity checks."""
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_keeps_highest_score():
    """The frame with the highest score is retained, regardless of order."""
    sel = BestFrameSelector()
    sel.observe(1, _frame(1), (0, 0, 4, 4), 0.5)
    sel.observe(1, _frame(2), (0, 0, 4, 4), 0.9)
    sel.observe(1, _frame(3), (0, 0, 4, 4), 0.7)

    best = sel.take(1)
    assert best is not None
    assert best.score == 0.9
    assert int(best.frame[0, 0, 0]) == 2


def test_take_removes_entry():
    """take() pops the candidate; a second take() returns None."""
    sel = BestFrameSelector()
    sel.observe(1, _frame(1), (0, 0, 4, 4), 0.5)
    assert sel.take(1) is not None
    assert sel.take(1) is None


def test_take_unknown_returns_none():
    """Taking a track that was never observed returns None."""
    assert BestFrameSelector().take(99) is None


def test_discard_frees_entry():
    """discard() drops the retained frame for a track."""
    sel = BestFrameSelector()
    sel.observe(2, _frame(1), (0, 0, 4, 4), 0.5)
    sel.discard(2)
    assert sel.take(2) is None


def test_discard_unknown_is_safe():
    """Discarding an unknown track does not raise."""
    BestFrameSelector().discard(123)


def test_tracks_are_independent():
    """Each track keeps its own best frame."""
    sel = BestFrameSelector()
    sel.observe(1, _frame(1), (0, 0, 4, 4), 0.3)
    sel.observe(2, _frame(2), (0, 0, 4, 4), 0.8)

    best1 = sel.take(1)
    best2 = sel.take(2)
    assert best1 is not None and best1.score == 0.3
    assert best2 is not None and best2.score == 0.8


def test_equal_score_keeps_earlier_frame():
    """A tie (not strictly greater) does not replace the incumbent."""
    sel = BestFrameSelector()
    sel.observe(1, _frame(1), (0, 0, 4, 4), 0.5)
    sel.observe(1, _frame(2), (0, 0, 4, 4), 0.5)

    best = sel.take(1)
    assert best is not None
    assert int(best.frame[0, 0, 0]) == 1


def test_stores_the_box():
    """The box associated with the best frame is retained alongside it."""
    sel = BestFrameSelector()
    sel.observe(1, _frame(1), (10, 20, 30, 40), 0.9)
    best = sel.take(1)
    assert best is not None
    assert best.box == (10, 20, 30, 40)
