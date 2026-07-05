"""Pure coordination helpers for the geolocation prior cache.

These functions hold no I/O so they can be unit-tested without a database or a
camera. The stateful side (persistence) lives in
:mod:`birdscanner.db.geo_store`; the startup orchestration that ties a model to a
store lives in :mod:`birdscanner.detector.geo`.

The eventual Bayesian update (``apply_geo_prior(likelihood, prior)``) will also
live here, but is deferred until the real geolocation model is transferred and
the classifier exposes its full probability vector.
"""

import hashlib
from datetime import datetime
from typing import Callable, Optional

import numpy as np

from birdscanner.ml.geolocation import GeolocationModel

# Weekly presence vectors are precomputed for a full year. ISO week 53 (which a
# few years carry) is folded onto 52 by ``current_week`` so exactly this many
# vectors always cover the year.
WEEKS_IN_YEAR = 52


def current_week() -> int:
    """Return the current ISO week, clamped to ``[1, 52]``.

    A handful of years contain an ISO week 53; the cache only stores 52 vectors,
    so week 53 is folded onto 52 (its nearest neighbour).

    Returns:
        The current ISO week number in ``[1, 52]``.
    """
    return min(datetime.now().isocalendar()[1], WEEKS_IN_YEAR)


def species_signature(species_order: list[str]) -> str:
    """Return a stable hash of an ordered species list.

    Stored alongside the cached priors so that a changed classifier class set (or
    a reordering) is detected on startup and forces regeneration — the cached
    vectors are only valid while they stay aligned to the classifier's indices.

    Args:
        species_order: Species names in classifier index order.

    Returns:
        A hex SHA-256 digest of the ordered list.
    """
    joined = "\n".join(species_order)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def generate_all_weeks(
    model: GeolocationModel,
    latitude: float,
    longitude: float,
    on_progress: Optional[Callable[[int], None]] = None,
) -> dict[int, np.ndarray]:
    """Compute the presence vector for every week of the year.

    Args:
        model: The geolocation model to query.
        latitude: Location latitude in degrees.
        longitude: Location longitude in degrees.
        on_progress: Optional callback invoked with the week number (1..52) after
            each week is computed, so the caller can log progress.

    Returns:
        A mapping of ISO week (1..52) to its ``float32`` presence vector.
    """
    vectors: dict[int, np.ndarray] = {}
    for week in range(1, WEEKS_IN_YEAR + 1):
        vectors[week] = model.predict_week(latitude, longitude, week)
        if on_progress is not None:
            on_progress(week)
    return vectors
