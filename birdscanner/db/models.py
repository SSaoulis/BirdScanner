"""SQLModel ORM models for the BirdFinder database."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class DetectionRecord(SQLModel, table=True):
    """Persisted record for a single high-confidence bird detection.

    Columns:
        id: Auto-incrementing primary key.
        timestamp: Wall-clock time of the detection.
        species: Classified species name.
        confidence: Species-classification confidence (ConvNeXt) in [0, 1].
        detection_confidence: Object-detection confidence (YOLO11n on the IMX500)
            for the bird bounding box, in [0, 1] (nullable for legacy rows written
            before the object-detection score was persisted).
        image_path: Path to the full saved image, relative to IMAGE_DIR.
        thumbnail_path: Path to the 200x200 JPEG thumbnail, relative to IMAGE_DIR.
        video_path: Path to the saved short mp4 clip, relative to IMAGE_DIR
            (nullable for legacy rows and rows written before the clip finishes
            encoding — the file appears a few seconds after the row).
        no_video_reason: Why this detection has no clip, when ``video_path`` is
            NULL. One of ``"recorder_busy"`` (the single-flight recorder was
            already capturing another sighting's clip) or ``"disabled"`` (video
            recording is turned off). NULL when a clip exists, or for legacy rows
            written before the reason was persisted.
        track_id: Identifier from the stable-detection tracker (nullable for legacy writes).
        stable_frames: Number of consecutive frames the track was stable before classification.
        duration_sec: Approximate track lifetime in seconds (nullable if unavailable).
        uploaded_at: Timestamp of cloud upload; NULL means not yet uploaded.
        box_x: Detection box left edge as a fraction [0, 1] of the saved image width
            (nullable for legacy rows written before boxes were persisted).
        box_y: Detection box top edge as a fraction [0, 1] of the saved image height.
        box_w: Detection box width as a fraction [0, 1] of the saved image width.
        box_h: Detection box height as a fraction [0, 1] of the saved image height.
        classifier_species: The classifier's own top class *before* the geomodel
            Bayesian update, when the update ran. NULL when no geomodel prior was
            applied (no location configured / prior unavailable) and for legacy
            rows; in that case ``species`` is the classifier's unadjusted pick.
        classifier_confidence: The classifier's softmax confidence for
            ``classifier_species``, in [0, 1] (NULL under the same conditions).
        geo_scores: JSON array of the top pre-normalised ``[species, score]`` pairs
            (``p(y|x)·p(y|c)`` before renormalising) from the geomodel update, kept
            for debugging/inspection. NULL when the update did not run / legacy rows.
        classifier_scores: JSON array of the classifier's own top-k ``[species,
            probability]`` softmax pairs (the raw distribution *before* any geomodel
            reweighting), kept for the Advanced-stats panel. NULL for legacy rows and
            when the classifier could not produce a distribution (test fakes).
        corrected: ``True`` when a user manually overrode the classifier's species
            (see ``birdscanner/db/corrector.py``). NULL/``False`` means the species
            is the model's own top prediction. When set, ``species`` is the
            human-chosen label and ``original_species`` holds the model's guess.
        original_species: The classifier's original top-1 species, preserved when a
            user corrects the detection so the model-vs-human disagreement stays on
            record (retraining ground truth). ``confidence`` is that guess's score.
            NULL when the detection was never corrected.
    """

    __tablename__ = "detections"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime
    species: str
    confidence: float
    detection_confidence: Optional[float] = Field(default=None, nullable=True)
    image_path: str
    thumbnail_path: str
    video_path: Optional[str] = Field(default=None, nullable=True)
    no_video_reason: Optional[str] = Field(default=None, nullable=True)
    track_id: Optional[int] = Field(default=None, nullable=True)
    stable_frames: Optional[int] = Field(default=None, nullable=True)
    duration_sec: Optional[float] = Field(default=None, nullable=True)
    uploaded_at: Optional[datetime] = Field(default=None, nullable=True)
    box_x: Optional[float] = Field(default=None, nullable=True)
    box_y: Optional[float] = Field(default=None, nullable=True)
    box_w: Optional[float] = Field(default=None, nullable=True)
    box_h: Optional[float] = Field(default=None, nullable=True)
    classifier_species: Optional[str] = Field(default=None, nullable=True)
    classifier_confidence: Optional[float] = Field(default=None, nullable=True)
    geo_scores: Optional[str] = Field(default=None, nullable=True)
    classifier_scores: Optional[str] = Field(default=None, nullable=True)
    corrected: Optional[bool] = Field(default=None, nullable=True)
    original_species: Optional[str] = Field(default=None, nullable=True)


class GeoPrior(SQLModel, table=True):
    """One cell of the geomodel spatio-temporal species prior.

    The BirdNET geomodel yields, for the configured location, a per-species
    probability of occurrence for each of the 48 "weeks" of the year. Only the
    species the classifier can predict are kept (projected via the geomodel<->
    classifier crosswalk), so the table holds ``n_classifier_species * 48`` rows.
    It is rebuilt on startup whenever the configured location changes
    (see :mod:`birdscanner.db.geo_prior_store`).

    Columns:
        id: Auto-incrementing primary key.
        species: Classifier class label the prior applies to.
        week: Week of the year, 1..48 (the geomodel's temporal resolution).
        probability: Occurrence prior in [0, 1] (the geomodel's sigmoid output).
    """

    __tablename__ = "geo_priors"  # type: ignore[assignment]

    id: Optional[int] = Field(default=None, primary_key=True)
    species: str = Field(index=True)
    week: int = Field(index=True)
    probability: float


class GeoPriorMeta(SQLModel, table=True):
    """Single-row record of the location the geo priors were last computed for.

    Startup compares the configured latitude/longitude against this row to decide
    whether the :class:`GeoPrior` table is stale and must be rebuilt. A single row
    (``id == 1``) is kept — it is upserted alongside every prior rebuild.

    Columns:
        id: Primary key; always ``1`` (only one row is ever stored).
        latitude: Latitude the priors were computed for, in degrees.
        longitude: Longitude the priors were computed for, in degrees.
        generated_at: Wall-clock time the priors were last (re)built.
    """

    __tablename__ = "geo_prior_meta"  # type: ignore[assignment]

    id: Optional[int] = Field(default=1, primary_key=True)
    latitude: float
    longitude: float
    generated_at: datetime


class CustomSpecies(SQLModel, table=True):
    """A user-added species label not present in the classifier's vocabulary.

    When a bird the classifier was never trained on is seen, the user can add its
    species as a manual correction from the Lightbox (see
    :mod:`birdscanner.db.custom_species`). The label is stored here so it persists
    across restarts and reappears in the correction picker: the detector serves
    the union of the classifier's classes and these rows as the species vocabulary
    (see the control server's ``GET /labels``). A custom-labelled detection is an
    ordinary ``corrected`` row in :class:`DetectionRecord`; this table only records
    the *label* so it can be offered again.

    Columns:
        name: The species label, stored exact-cased and used as the primary key
            (so a label is stored at most once). Case-insensitive de-duplication is
            enforced by :func:`birdscanner.db.custom_species.add_custom_species`.
        created_at: Wall-clock time the label was first added.
    """

    __tablename__ = "custom_species"  # type: ignore[assignment]

    name: str = Field(primary_key=True)
    created_at: datetime
