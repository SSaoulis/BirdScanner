"""Runtime-editable settings: domain, validation, and JSON persistence.

The static :mod:`birdscanner.detector.config.config` holds the *code* defaults for the
detector.  This module layers a user-editable overlay on top of it, persisted as
JSON on the detector's writable data volume (mirroring
:mod:`birdscanner.detector.hardware.crop`), so the Settings page in the UI can change
detection parameters without editing code or rebuilding the container.

Only a subset of settings can be applied to a *running* detector — the detection
confidence threshold, the save-side classification floor, the ignore list, and
the log level are read live, so they take effect immediately (see
:mod:`birdscanner.detector.config.settings_controller`).  The rest are consumed once at
startup wiring (camera model, stability duration, video, save location,
threading), so they are persisted and take effect on the next detector boot;
:data:`RESTART_FIELDS` records which those are so the UI can flag them.

The pure domain lives here so it can be unit-tested without a camera; the
application of live changes to the running pipeline lives in
``settings_controller.py`` (as ``crop_controller.py`` is to ``crop.py``).
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional

from birdscanner.detector.config.config import config as app_config
from birdscanner.ml.classification_pipeline import (
    DEFAULT_SAVE_CONFIDENCE_THRESHOLD,
    IMAGE_DIR,
)

logger = logging.getLogger("tracking")


@dataclass
class Settings:
    """User-editable detector settings persisted as JSON.

    Attributes:
        detection_threshold: Minimum YOLO object-detection confidence to keep a
            detection, in ``[0, 1]`` (applied live).
        classification_threshold: Minimum ConvNeXt species-classification
            confidence before a detection is saved/persisted, in ``[0, 1]``
            (applied live).
        ignore_species: Species names that are never saved even when classified
            (matched case-insensitively; applied live).
        excluded_classes: Object-detection (YOLO/COCO) class labels dropped
            before tracking, so false positives on unwanted classes (e.g.
            ``"bench"``) never enter the tracker/logs (matched
            case-insensitively; applied live).
        stability_seconds: Seconds a track must be stable before classification
            fires (``config.object_duration_threshold``; requires restart).
        image_dir: Root directory saved images/clips are written to (requires
            restart; must stay under the shared data volume so the API can serve
            the files).
        video_save: Whether to save a short mp4 clip per detection (restart).
        video_pre_roll_seconds: Seconds of buffered footage prepended to a clip
            (restart).
        video_post_roll_seconds: Seconds recorded after a clip triggers
            (restart).
        multithread: Run classification on a background thread (restart).
        debug: Enable DEBUG-level tracking logs (applied live).
        latitude: Deployment latitude in degrees for the geomodel prior, or
            ``None`` when no location is set (the prior is then not built).
            Rebuilding the prior happens at startup, so this requires a restart.
        longitude: Deployment longitude in degrees for the geomodel prior, or
            ``None`` when no location is set (requires restart).
    """

    detection_threshold: float
    classification_threshold: float
    ignore_species: list[str]
    excluded_classes: list[str]
    stability_seconds: float
    image_dir: str
    video_save: bool
    video_pre_roll_seconds: float
    video_post_roll_seconds: float
    multithread: bool
    debug: bool
    latitude: Optional[float]
    longitude: Optional[float]


# Fields that a running detector reads live, so a change takes effect at once.
# Everything else is consumed at startup wiring and needs a detector restart.
LIVE_FIELDS: frozenset[str] = frozenset(
    {
        "detection_threshold",
        "classification_threshold",
        "ignore_species",
        "excluded_classes",
        "debug",
    }
)

# Fields that only take effect on the next detector boot (used by the UI to flag
# a pending restart).  Derived so it can never drift out of sync with the model.
RESTART_FIELDS: frozenset[str] = (
    frozenset(Settings.__dataclass_fields__) - LIVE_FIELDS  # pylint: disable=no-member
)

# Validation buckets by field name.
_FLOAT01_FIELDS = frozenset({"detection_threshold", "classification_threshold"})
_POSITIVE_FLOAT_FIELDS = frozenset(
    {"stability_seconds", "video_pre_roll_seconds", "video_post_roll_seconds"}
)
_BOOL_FIELDS = frozenset({"video_save", "multithread", "debug"})
_STRING_LIST_FIELDS = frozenset({"ignore_species", "excluded_classes"})
# Optional coordinate fields: None (unset) is allowed, else within (lo, hi) degrees.
_RANGED_OPTIONAL_FLOAT_FIELDS: dict[str, tuple[float, float]] = {
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
}


def default_settings() -> Settings:
    """Return the settings seeded from the static config + environment.

    Used when no settings file exists yet, so an existing deployment keeps its
    current ``config`` defaults and ``IMAGE_DIR`` until the user changes them.

    Returns:
        A :class:`Settings` populated from :mod:`config` and the environment.
    """
    return Settings(
        detection_threshold=app_config.threshold,
        classification_threshold=DEFAULT_SAVE_CONFIDENCE_THRESHOLD,
        ignore_species=[],
        excluded_classes=sorted(app_config.excluded_classes),
        stability_seconds=app_config.object_duration_threshold,
        image_dir=IMAGE_DIR,
        video_save=app_config.video.save,
        video_pre_roll_seconds=app_config.video.pre_roll_seconds,
        video_post_roll_seconds=app_config.video.post_roll_seconds,
        multithread=app_config.multithread,
        debug=app_config.debug,
        latitude=app_config.latitude,
        longitude=app_config.longitude,
    )


def _as_float01(name: str, value: Any) -> float:
    """Coerce ``value`` to a float in ``[0, 1]`` or raise ``ValueError``."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _as_positive_float(name: str, value: Any) -> float:
    """Coerce ``value`` to a non-negative float or raise ``ValueError``."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if number < 0.0:
        raise ValueError(f"{name} must not be negative")
    return number


def _as_bool(name: str, value: Any) -> bool:
    """Coerce ``value`` to a bool or raise ``ValueError``."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _as_optional_ranged_float(
    name: str, value: Any, lo: float, hi: float
) -> Optional[float]:
    """Coerce ``value`` to a float in ``[lo, hi]``, allowing ``None`` (unset).

    ``None`` means the coordinate is not set (the geomodel prior is then not built).
    Any other value must parse as a number within the inclusive range.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number or null") from exc
    if not lo <= number <= hi:
        raise ValueError(f"{name} must be between {lo:g} and {hi:g}")
    return number


def _as_string_list(name: str, value: Any) -> list[str]:
    """Coerce ``value`` to a de-duplicated list of non-empty, trimmed strings."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    seen: dict[str, None] = {}
    for item in value:
        trimmed = item.strip()
        if trimmed:
            seen.setdefault(trimmed, None)
    return list(seen)


def _coerce_field(name: str, value: Any) -> Any:
    """Validate + coerce one field's incoming value by its name."""
    if name in _FLOAT01_FIELDS:
        return _as_float01(name, value)
    if name in _POSITIVE_FLOAT_FIELDS:
        return _as_positive_float(name, value)
    if name in _BOOL_FIELDS:
        return _as_bool(name, value)
    if name in _RANGED_OPTIONAL_FLOAT_FIELDS:
        lo, hi = _RANGED_OPTIONAL_FLOAT_FIELDS[name]
        return _as_optional_ranged_float(name, value, lo, hi)
    if name in _STRING_LIST_FIELDS:
        return _as_string_list(name, value)
    if name == "image_dir":
        text = str(value).strip()
        if not text:
            raise ValueError("image_dir must not be empty")
        return text
    raise ValueError(f"Unknown setting: {name}")


def merge_settings(current: Settings, updates: dict[str, Any]) -> Settings:
    """Return ``current`` with validated ``updates`` applied.

    Only known fields may be updated; each value is validated/coerced by its
    field type.  Unknown keys and out-of-range values raise ``ValueError`` so the
    caller can return a 400 without mutating any state.

    Args:
        current: The settings to update.
        updates: A partial mapping of field name -> new value.

    Returns:
        A new :class:`Settings` with the updates applied.

    Raises:
        ValueError: If a key is unknown or a value fails validation.
    """
    known = set(Settings.__dataclass_fields__)  # pylint: disable=no-member
    coerced = {}
    for name, value in updates.items():
        if name not in known:
            raise ValueError(f"Unknown setting: {name}")
        coerced[name] = _coerce_field(name, value)
    return replace(current, **coerced)


def settings_config_path() -> str:
    """Return the path of the settings JSON file.

    Reads ``SETTINGS_PATH``; defaults to ``/data/settings.json`` (the detector's
    writable data volume, alongside ``crop.json``).

    Returns:
        The filesystem path to persist/read the settings overlay.
    """
    return os.environ.get("SETTINGS_PATH", "/data/settings.json")


def load_settings(path: str) -> Settings:
    """Load the persisted settings, falling back to defaults on any problem.

    A missing file, malformed JSON, or an out-of-range value all yield the
    defaults (with any *valid* fields still applied), so a corrupt file can never
    stop the detector from starting.

    Args:
        path: Path to the JSON file written by :func:`save_settings`.

    Returns:
        The loaded :class:`Settings`, or the defaults when unreadable.
    """
    defaults = default_settings()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return defaults
    except (OSError, ValueError) as exc:
        logger.warning("Could not read settings %s (%s); using defaults", path, exc)
        return defaults
    if not isinstance(data, dict):
        logger.warning("Settings file %s is not an object; using defaults", path)
        return defaults
    try:
        return merge_settings(defaults, data)
    except ValueError as exc:
        logger.warning("Invalid settings in %s (%s); using defaults", path, exc)
        return defaults


def save_settings(path: str, settings: Settings) -> None:
    """Persist settings to JSON atomically.

    Writes to a temporary sibling file and renames it into place so a concurrent
    reader never sees a half-written file.

    Args:
        path: Destination JSON path.
        settings: The settings to persist.

    Raises:
        OSError: If the file cannot be written (the caller surfaces this so the
            UI can report that the change was not saved).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(asdict(settings), handle, indent=2)
    os.replace(tmp, path)
