"""Separate SQLite store for precomputed geolocation prior vectors.

The geolocation model's per-species presence vectors depend only on location +
week, so they are computed once and cached rather than recomputed per detection
(see :mod:`birdscanner.ml.geolocation`). They live in their own database file
(``GEO_DB_PATH``, default ``/data/geo_priors.db``) — separate from the detections
DB — so this store owns its **own** SQLAlchemy ``MetaData`` and never
cross-creates tables with :mod:`birdscanner.db.models` (whose ``init_db`` on the
detections DB must not touch these tables, and vice-versa).

Schema (per the "row per week" layout):

* ``geo_meta``       — a single row: the location + species order + signature the
  cached vectors were generated for (checked on startup to decide regeneration).
* ``geo_week_prior`` — 52 rows keyed by ISO week; each ``vector`` is the presence
  probability vector stored as a compact ``float32`` blob.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    delete,
    insert,
    select,
)

from birdscanner.db.database import make_engine

_DEFAULT_GEO_DB_PATH = "/data/geo_priors.db"

# A dedicated metadata registry keeps these tables off SQLModel's shared metadata
# so the detections DB and the geo DB never create each other's tables.
_GEO_METADATA = MetaData()

geo_meta_table = Table(
    "geo_meta",
    _GEO_METADATA,
    Column("id", Integer, primary_key=True),
    Column("latitude", Float, nullable=False),
    Column("longitude", Float, nullable=False),
    Column("species_order", Text, nullable=False),
    Column("species_signature", String, nullable=False),
    Column("model_version", String, nullable=False),
    Column("generated_at", DateTime, nullable=False),
)

geo_week_prior_table = Table(
    "geo_week_prior",
    _GEO_METADATA,
    Column("week", Integer, primary_key=True),
    Column("vector", LargeBinary, nullable=False),
)


@dataclass
class GeoMeta:
    """Metadata describing what a cached set of geolocation priors was built for.

    Attributes:
        latitude: Location latitude the vectors were generated for.
        longitude: Location longitude the vectors were generated for.
        species_order: Species names (classifier index order) the vectors align to.
        species_signature: Hash of ``species_order`` (see
            :func:`birdscanner.ml.geo_priors.species_signature`); a mismatch on
            startup forces regeneration.
        model_version: The geolocation model's version tag; a change forces
            regeneration.
        generated_at: When the vectors were computed.
    """

    latitude: float
    longitude: float
    species_order: list[str]
    species_signature: str
    model_version: str
    generated_at: datetime


def geo_db_path() -> str:
    """Return the geo-prior database path.

    Reads ``GEO_DB_PATH``; defaults to ``/data/geo_priors.db`` (the detector's
    writable data volume, alongside ``detections.db`` / ``crop.json`` /
    ``settings.json``).

    Returns:
        The filesystem path to the geo-prior SQLite database.
    """
    return os.environ.get("GEO_DB_PATH", _DEFAULT_GEO_DB_PATH)


def _vector_to_blob(vector: np.ndarray) -> bytes:
    """Serialise a probability vector to a compact ``float32`` byte blob."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def _blob_to_vector(blob: bytes) -> np.ndarray:
    """Deserialise a ``float32`` byte blob back into a writable numpy vector."""
    # frombuffer yields a read-only view over the bytes; copy so callers can write.
    return np.frombuffer(blob, dtype=np.float32).copy()


class GeoPriorStore:
    """Read/write access to the cached geolocation prior vectors.

    Owned by the detector (which holds the read-write data volume). Creating the
    store also creates its two tables if they do not exist.
    """

    def __init__(self, db_path: str) -> None:
        """Open (creating if needed) the geo-prior database.

        Args:
            db_path: Filesystem path to the SQLite database file.
        """
        self._engine = make_engine(db_path)
        _GEO_METADATA.create_all(self._engine)

    def read_meta(self) -> Optional[GeoMeta]:
        """Return the stored location metadata, or ``None`` when the cache is empty.

        Returns:
            The single :class:`GeoMeta` row, or ``None`` if nothing is cached yet.
        """
        with self._engine.connect() as conn:
            row = conn.execute(select(geo_meta_table)).first()
        if row is None:
            return None
        return GeoMeta(
            latitude=row.latitude,
            longitude=row.longitude,
            species_order=json.loads(row.species_order),
            species_signature=row.species_signature,
            model_version=row.model_version,
            generated_at=row.generated_at,
        )

    def write_priors(self, meta: GeoMeta, vectors: dict[int, np.ndarray]) -> None:
        """Replace the cached metadata and all week vectors in one transaction.

        Clears any previous cache first so a re-generation for a new location
        never leaves stale weeks behind.

        Args:
            meta: The metadata describing this set of vectors.
            vectors: Mapping of ISO week (1..52) to its presence vector.
        """
        with self._engine.begin() as conn:
            conn.execute(delete(geo_week_prior_table))
            conn.execute(delete(geo_meta_table))
            conn.execute(
                insert(geo_meta_table).values(
                    latitude=meta.latitude,
                    longitude=meta.longitude,
                    species_order=json.dumps(meta.species_order),
                    species_signature=meta.species_signature,
                    model_version=meta.model_version,
                    generated_at=meta.generated_at,
                )
            )
            conn.execute(
                insert(geo_week_prior_table),
                [
                    {"week": week, "vector": _vector_to_blob(vector)}
                    for week, vector in sorted(vectors.items())
                ],
            )

    def get_week_vector(self, week: int) -> Optional[np.ndarray]:
        """Return the cached presence vector for one ISO week, or ``None``.

        Args:
            week: ISO week number (1..52).

        Returns:
            The ``float32`` presence vector, or ``None`` when that week (or the
            whole cache) is absent.
        """
        with self._engine.connect() as conn:
            row = conn.execute(
                select(geo_week_prior_table.c.vector).where(
                    geo_week_prior_table.c.week == week
                )
            ).first()
        if row is None:
            return None
        return _blob_to_vector(row.vector)

    def dispose(self) -> None:
        """Dispose the underlying engine (release the connection pool)."""
        self._engine.dispose()
