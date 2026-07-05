"""Startup coordination for the geolocation prior cache (mirrors ``gating.py``).

At startup the detector precomputes the geolocation model's 52 weekly presence
vectors for the configured location and caches them in a separate database (see
:mod:`birdscanner.db.geo_store`). Because the vectors depend only on location +
week, this runs **once** per location: on boot the stored location/species
signature is compared to the current configuration, and generation only fires
when they differ (a normal boot is a cache hit and does no work).

This bridges the pure model (:mod:`birdscanner.ml.geolocation`) and the pure
coordination helpers (:mod:`birdscanner.ml.geo_priors`) to the persistent store,
keeping the ``detector -> ml -> db`` layering intact.
"""

import logging
from datetime import datetime

from birdscanner.db.geo_store import GeoMeta, GeoPriorStore, geo_db_path
from birdscanner.detector.settings import Settings
from birdscanner.ml.geo_priors import generate_all_weeks, species_signature
from birdscanner.ml.geolocation import GeolocationModel

logger = logging.getLogger("tracking")

# Log a progress line every this many weeks while regenerating.
_PROGRESS_EVERY = 5


class GeoPriorProvider:
    """Current-week lookup over the cached geolocation priors.

    A thin read-only façade over :class:`GeoPriorStore` handed to the pipeline so
    that, once the Bayesian prior update is wired (a follow-up), a classification
    can fetch the current week's presence vector without touching the store's
    schema directly.
    """

    def __init__(self, store: GeoPriorStore) -> None:
        """Initialise the provider over an open store.

        Args:
            store: The geo-prior store to read cached vectors from.
        """
        self._store = store

    def week_vector(self, week: int):
        """Return the cached presence vector for an ISO week, or ``None``.

        Args:
            week: ISO week number (1..52).

        Returns:
            The ``float32`` presence vector, or ``None`` when absent.
        """
        return self._store.get_week_vector(week)


def _log_progress(week: int) -> None:
    """Log a generation-progress line every ``_PROGRESS_EVERY`` weeks (and at 52)."""
    if week % _PROGRESS_EVERY == 0 or week == 52:
        logger.info("Generated geo priors: %d/52 weeks", week)


def _needs_regeneration(
    meta: GeoMeta | None, latitude: float, longitude: float, signature: str
) -> bool:
    """Return whether the cache must be regenerated for the current inputs.

    Args:
        meta: The stored metadata, or ``None`` when nothing is cached.
        latitude: The configured location latitude.
        longitude: The configured location longitude.
        signature: The current species signature.

    Returns:
        ``True`` when there is no cache, or the location / species signature
        differs from what was cached.
    """
    if meta is None:
        return True
    return (
        meta.latitude != latitude
        or meta.longitude != longitude
        or meta.species_signature != signature
    )


def bootstrap_geo_priors(
    settings: Settings, model: GeolocationModel
) -> GeoPriorProvider:
    """Ensure the geolocation prior cache matches the configured location.

    Opens the geo-prior store, compares the cached ``(latitude, longitude,
    species_signature)`` to the current configuration, and regenerates all 52
    weekly vectors when they differ (logging progress every 5 weeks). On a match
    the existing cache is reused.

    Args:
        settings: The loaded settings (source of the configured location).
        model: The geolocation model to query.

    Returns:
        A :class:`GeoPriorProvider` over the (now up-to-date) store.
    """
    store = GeoPriorStore(geo_db_path())
    signature = species_signature(model.species_order)
    latitude, longitude = settings.latitude, settings.longitude
    meta = store.read_meta()

    if _needs_regeneration(meta, latitude, longitude, signature):
        logger.info(
            "Regenerating geo priors for lat=%.5f lon=%.5f (52 weeks)...",
            latitude,
            longitude,
        )
        vectors = generate_all_weeks(
            model, latitude, longitude, on_progress=_log_progress
        )
        store.write_priors(
            GeoMeta(
                latitude=latitude,
                longitude=longitude,
                species_order=list(model.species_order),
                species_signature=signature,
                model_version=model.model_version,
                generated_at=datetime.now(),
            ),
            vectors,
        )
        logger.info("Geo priors ready (%d weeks cached).", len(vectors))
    else:
        logger.info(
            "Geo priors up to date for lat=%.5f lon=%.5f; using cache.",
            latitude,
            longitude,
        )

    return GeoPriorProvider(store)
