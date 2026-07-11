"""Startup orchestration for the geomodel spatio-temporal species prior.

On boot the detector computes the BirdNET geomodel's per-species occurrence prior
for the configured location, keeps only the classifier's species (via the label
crosswalk), and persists the resulting ``48-week x ~706-species`` matrix to the DB
(see :mod:`birdscanner.db.geo_prior_store`). The rebuild is skipped when the stored
location already matches the settings, so a normal restart is a cheap no-op.

The pure geomodel inference + projection lives in
:mod:`birdscanner.ml.geomodel`; the persistence in
:mod:`birdscanner.db.geo_prior_store`. This module only wires them together and
must never crash startup — a missing geomodel model file or an inference error is
logged and swallowed (mirroring ``camera.wait_for_camera`` and the classifier-absent
skips), so the detector keeps running with whatever prior (if any) is already stored.
"""

import logging
from typing import Callable, Optional

from birdscanner.db.database import SessionFactory
from birdscanner.db.geo_prior_store import (
    has_geo_priors,
    location_matches,
    read_meta,
    replace_geo_priors,
)
from birdscanner.detector.paths import (
    geomodel_labels_path,
    geomodel_map_path,
    geomodel_model_path,
)
from birdscanner.ml.geomodel import compute_classifier_priors

logger = logging.getLogger("tracking")

# Signature of the injectable prior-computation function (defaults to the real
# geomodel inference); overridden in tests so they never load the 28 MB ONNX model.
ComputePriors = Callable[[float, float], dict[str, list[float]]]


def _default_compute(lat: float, lon: float) -> dict[str, list[float]]:
    """Compute the classifier-aligned geo prior via the real geomodel + crosswalk."""
    return compute_classifier_priors(
        str(geomodel_model_path()),
        str(geomodel_labels_path()),
        str(geomodel_map_path()),
        lat,
        lon,
    )


def refresh_geo_priors(
    session_factory: SessionFactory,
    lat: Optional[float],
    lon: Optional[float],
    *,
    compute: ComputePriors = _default_compute,
) -> bool:
    """Rebuild the stored geo prior when the configured location has changed.

    Called once at startup. Does nothing when no location is set, or when the
    stored priors already cover the configured location; otherwise computes the
    prior for ``(lat, lon)`` and replaces the stored rows. Any failure to compute
    (e.g. the geomodel model file is absent) is logged and swallowed so the
    detector still starts.

    Args:
        session_factory: Zero-argument callable returning a ``Session`` context
            manager (see :func:`db.database.make_session_factory`).
        lat: The configured latitude in degrees, or ``None`` when unset.
        lon: The configured longitude in degrees, or ``None`` when unset.
        compute: Function mapping ``(lat, lon)`` to
            ``{species: [48 weekly probabilities]}``; injectable for testing.

    Returns:
        ``True`` when the prior was (re)built, ``False`` when it was skipped
        (no location, already current, or a computation error).
    """
    if lat is None or lon is None:
        logger.info("Geo prior skipped: no location set.")
        return False

    meta = read_meta(session_factory)
    if location_matches(meta, lat, lon) and has_geo_priors(session_factory):
        logger.info(
            "Geo prior already current for lat=%.5f lon=%.5f; skipping rebuild.",
            lat,
            lon,
        )
        return False

    logger.info("Building geo prior for lat=%.5f lon=%.5f ...", lat, lon)
    try:
        priors = compute(lat, lon)
    except Exception as exc:  # noqa: BLE001 — never crash startup on a prior failure
        logger.warning(
            "Could not build geo prior for lat=%.5f lon=%.5f (%s); "
            "continuing without an updated prior.",
            lat,
            lon,
            exc,
        )
        return False

    if not priors:
        logger.warning("Geo prior computation returned no species; nothing stored.")
        return False

    replace_geo_priors(session_factory, lat, lon, priors)
    return True
