"""Persistence for the geomodel spatio-temporal species prior.

The detector process owns the database read-write, so the geo prior is (re)built
here, synchronously, at startup (the API mounts the DB read-only). The store keeps
the ``geo_priors`` table (one row per classifier species per week) and a single-row
``geo_prior_meta`` recording the location those priors were computed for, so a
restart can tell whether the priors are stale and must be rebuilt.

The pure geomodel inference + projection lives in :mod:`birdscanner.ml.geomodel`;
this module only reads and writes the database.
"""

import logging
from datetime import datetime

from sqlmodel import delete, select

from birdscanner.db.database import SessionFactory
from birdscanner.db.models import GeoPrior, GeoPriorMeta

logger = logging.getLogger("tracking")

# Latitude/longitude tolerance (degrees) below which a location is unchanged, so a
# rebuild is skipped. ~1e-4 deg is ~11 m — far finer than the geomodel's resolution.
LOCATION_MATCH_TOLERANCE = 1e-4

# Fixed primary key of the single metadata row.
_META_ID = 1


def read_meta(session_factory: SessionFactory) -> GeoPriorMeta | None:
    """Return the stored geo-prior metadata row, or ``None`` if never built.

    Args:
        session_factory: Zero-argument callable returning a ``Session`` context
            manager (see :func:`db.database.make_session_factory`).

    Returns:
        The single :class:`GeoPriorMeta` row, or ``None`` when the priors have
        not been built yet.
    """
    with session_factory() as session:
        return session.get(GeoPriorMeta, _META_ID)


def location_matches(
    meta: GeoPriorMeta | None,
    lat: float,
    lon: float,
    tol: float = LOCATION_MATCH_TOLERANCE,
) -> bool:
    """Report whether stored priors already cover the given location.

    Args:
        meta: The stored metadata row (or ``None`` when nothing is stored).
        lat: The configured latitude, in degrees.
        lon: The configured longitude, in degrees.
        tol: Absolute per-axis tolerance in degrees; differences smaller than
            this are treated as the same location.

    Returns:
        ``True`` when ``meta`` exists and both coordinates are within ``tol`` of
        the requested location; ``False`` otherwise.
    """
    if meta is None:
        return False
    return abs(meta.latitude - lat) <= tol and abs(meta.longitude - lon) <= tol


def replace_geo_priors(
    session_factory: SessionFactory,
    lat: float,
    lon: float,
    priors: dict[str, list[float]],
) -> int:
    """Replace all stored geo priors with a freshly computed set.

    In a single transaction: delete every existing :class:`GeoPrior` row, insert one
    row per ``(species, week)`` from ``priors`` (weeks numbered ``1..len``), and
    upsert the single :class:`GeoPriorMeta` row to record the location and time.

    Args:
        session_factory: Zero-argument callable returning a ``Session`` context
            manager.
        lat: Latitude the priors were computed for, in degrees.
        lon: Longitude the priors were computed for, in degrees.
        priors: ``{species: [prob_week_1, ...]}`` from
            :func:`birdscanner.ml.geomodel.project_to_classifier`.

    Returns:
        The number of :class:`GeoPrior` rows written.
    """
    with session_factory() as session:
        session.exec(delete(GeoPrior))  # type: ignore[call-overload]

        rows = [
            GeoPrior(species=species, week=week, probability=probability)
            for species, weekly in priors.items()
            for week, probability in enumerate(weekly, start=1)
        ]
        session.add_all(rows)

        meta = session.get(GeoPriorMeta, _META_ID)
        if meta is None:
            session.add(
                GeoPriorMeta(
                    id=_META_ID,
                    latitude=lat,
                    longitude=lon,
                    generated_at=datetime.now(),
                )
            )
        else:
            meta.latitude = lat
            meta.longitude = lon
            meta.generated_at = datetime.now()
            session.add(meta)

        session.commit()

    logger.info(
        "Stored %d geo-prior rows (%d species) for lat=%.5f lon=%.5f",
        len(rows),
        len(priors),
        lat,
        lon,
    )
    return len(rows)


def has_geo_priors(session_factory: SessionFactory) -> bool:
    """Report whether any :class:`GeoPrior` rows are present.

    Used alongside :func:`location_matches` so a matching location whose rows were
    somehow lost still triggers a rebuild.

    Args:
        session_factory: Zero-argument callable returning a ``Session`` context
            manager.

    Returns:
        ``True`` when at least one prior row exists.
    """
    with session_factory() as session:
        return session.exec(select(GeoPrior.id).limit(1)).first() is not None
