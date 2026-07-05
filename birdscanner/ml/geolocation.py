"""Geolocation presence model: the third-stage model seam.

The geolocation model maps ``(latitude, longitude, week_of_year)`` to a
probability vector over the classifier's species — the estimated presence
likelihood of each species at that place and time of year. Eventually this
vector becomes the *prior* in a Bayesian update against the classifier's softmax
*likelihood* (``posterior ∝ prior × likelihood``) to suppress out-of-range false
positives.

Because the output depends only on location + week, it is deterministic per
location and is precomputed once (all 52 weeks) and cached in a separate database
(see :mod:`birdscanner.db.geo_store`); the coordination that drives that
precompute lives in :mod:`birdscanner.detector.geo`.

The real model file has not been transferred yet, so this module ships a
:class:`PlaceholderGeolocationModel` that returns a uniform vector (a no-op prior
under a Bayesian update). Dropping in the real model later replaces only this
class — the store, generation, settings, and wiring are model-agnostic.

This module is pure ``ml/`` (no ``detector``/``api`` imports), so it stays
importable and unit-testable off the Pi.
"""

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GeolocationModel(Protocol):
    """A model producing per-species presence probabilities for a place/week.

    Implementations must expose the ``species_order`` their vectors are aligned
    to (so a stored prior can be matched to the classifier's class indices) and a
    ``model_version`` string (so a changed model forces the cache to regenerate).
    """

    @property
    def species_order(self) -> list[str]:
        """Species names, in the index order every returned vector is aligned to."""

    @property
    def model_version(self) -> str:
        """Opaque version tag; a change invalidates any cached priors."""

    def predict_week(self, latitude: float, longitude: float, week: int) -> np.ndarray:
        """Return the presence-probability vector for one ISO week (1..52).

        Args:
            latitude: Location latitude in degrees, ``[-90, 90]``.
            longitude: Location longitude in degrees, ``[-180, 180]``.
            week: ISO week of the year, ``1..52``.

        Returns:
            A 1-D ``float32`` vector aligned to :attr:`species_order`.
        """


class PlaceholderGeolocationModel:
    """A stand-in geolocation model used until the real one is transferred.

    Returns a uniform probability vector for every location and week, which is a
    no-op as a Bayesian prior (multiplying a likelihood by a constant and
    renormalising leaves it unchanged). It exists so the database, generation
    loop, settings, and startup wiring can be built, tested, and shipped ahead of
    the real model.
    """

    _MODEL_VERSION = "placeholder-1"

    def __init__(self, species_order: list[str]) -> None:
        """Initialise the placeholder over a fixed species order.

        Args:
            species_order: Species names (in classifier index order) the returned
                vectors are aligned to.

        Raises:
            ValueError: If ``species_order`` is empty.
        """
        if not species_order:
            raise ValueError("species_order must be non-empty")
        self._species_order = list(species_order)
        self._uniform = np.full(
            len(self._species_order), 1.0 / len(self._species_order), dtype=np.float32
        )

    @property
    def species_order(self) -> list[str]:
        """Species names, in the index order every returned vector is aligned to."""
        return list(self._species_order)

    @property
    def model_version(self) -> str:
        """Opaque version tag; a change invalidates any cached priors."""
        return self._MODEL_VERSION

    def predict_week(self, latitude: float, longitude: float, week: int) -> np.ndarray:
        """Return a uniform presence vector (placeholder for the real model).

        Args:
            latitude: Ignored by the placeholder.
            longitude: Ignored by the placeholder.
            week: Ignored by the placeholder.

        Returns:
            A copy of the uniform ``float32`` vector over :attr:`species_order`.
        """
        # The placeholder is location/week-independent (a uniform prior is a
        # Bayesian no-op); the real model will consume these arguments.
        # pylint: disable=unused-argument
        return self._uniform.copy()


def load_species_order(class_to_idx_path: str | Path) -> list[str]:
    """Load the classifier's species names in index order from its class map.

    The classifier stores a ``{class_name: index}`` JSON map; the geolocation
    vectors must be aligned to the same index order so a future element-wise
    prior × likelihood multiply lines the two up. Ordering by index yields that
    canonical species order.

    Args:
        class_to_idx_path: Path to the classifier's ``{class_name: index}`` JSON.

    Returns:
        Species names ordered by their classifier index.
    """
    with Path(class_to_idx_path).open("r", encoding="utf-8") as handle:
        class_to_idx: dict[str, int] = json.load(handle)
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
