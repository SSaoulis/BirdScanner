"""Apply persisted settings to the detector — at boot and live at runtime.

:mod:`birdscanner.detector.config.settings` holds the pure settings domain (the model,
validation, and JSON load/save).  This module is the *stateful* side that pushes
those values onto the running detector, mirroring how ``crop_controller.py``
applies ``crop.py`` to the live camera:

* :func:`apply_settings_to_config` runs once at startup, before the pipeline is
  wired, so every setting (live *and* restart-only) shapes the initial build.
* :class:`SettingsController` handles runtime updates: it validates + persists
  the change, applies the live-safe fields to the running pipeline immediately,
  and reports whether a restart-only field changed so the UI can prompt for a
  restart.
"""

import logging
from dataclasses import asdict
from typing import Any

from birdscanner.ml import classification_pipeline
from birdscanner.ml.classification_pipeline import PipelineContext
from birdscanner.detector.config.config import config as app_config
from birdscanner.detector.config.settings import (
    LIVE_FIELDS,
    RESTART_FIELDS,
    Settings,
    merge_settings,
    save_settings,
)

logger = logging.getLogger("tracking")


def _set_log_level(debug: bool) -> None:
    """Set the ``tracking`` logger (and its handlers) to DEBUG or INFO live."""
    level = logging.DEBUG if debug else logging.INFO
    tracking_logger = logging.getLogger("tracking")
    tracking_logger.setLevel(level)
    for handler in tracking_logger.handlers:
        handler.setLevel(level)


def apply_settings_to_config(settings: Settings) -> None:
    """Push every setting onto the static config, before the pipeline is built.

    Called once at startup so restart-only settings (stability duration, video,
    threading, save location) shape the initial wiring, and live settings seed
    their starting values.

    Args:
        settings: The settings loaded for this boot.
    """
    app_config.threshold = settings.detection_threshold
    app_config.included_classes = {name.lower() for name in settings.included_classes}
    app_config.object_duration_threshold = settings.stability_seconds
    app_config.multithread = settings.multithread
    app_config.debug = settings.debug
    app_config.video.save = settings.video_save
    app_config.video.pre_roll_seconds = settings.video_pre_roll_seconds
    app_config.video.post_roll_seconds = settings.video_post_roll_seconds
    app_config.latitude = settings.latitude
    app_config.longitude = settings.longitude
    # IMAGE_DIR is a module-level global read when each detection is saved.
    classification_pipeline.IMAGE_DIR = settings.image_dir


def apply_settings_to_context(settings: Settings, context: PipelineContext) -> None:
    """Apply the save-side classification settings to the pipeline context.

    These live on the :class:`PipelineContext` (not ``config``) because they are
    consumed inside ``ml/``, which must not import ``detector`` — so the detector
    injects them here instead.

    Args:
        settings: The current settings.
        context: The running pipeline's context to mutate.
    """
    context.save_confidence_threshold = settings.classification_threshold
    context.ignore_species = {name.lower() for name in settings.ignore_species}


class SettingsController:
    """Validate, persist, and live-apply runtime settings changes.

    Holds a reference to the running pipeline's :class:`PipelineContext` so the
    live-safe fields take effect immediately.  Restart-only fields are persisted
    and reported via :meth:`get_state` so the UI can prompt for a restart; they
    take effect on the next boot via :func:`apply_settings_to_config`.
    """

    def __init__(
        self, path: str, boot_settings: Settings, context: PipelineContext
    ) -> None:
        """Initialise the controller.

        Args:
            path: JSON file the settings are persisted to.
            boot_settings: The settings the detector started with (the baseline
                for detecting pending restart-only changes).
            context: The running pipeline context to apply live changes to.
        """
        self._path = path
        self._boot = boot_settings
        self._current = boot_settings
        self._context = context

    def get_state(self) -> dict[str, Any]:
        """Return the current settings plus restart metadata for the UI.

        Returns:
            A dict with ``settings`` (the current values), ``needs_restart``
            (whether a restart-only field differs from the booted value),
            ``restart_fields`` and ``live_fields`` (so the UI can badge each
            field without duplicating the classification).
        """
        return {
            "settings": asdict(self._current),
            "needs_restart": self._needs_restart(),
            "restart_fields": sorted(RESTART_FIELDS),
            "live_fields": sorted(LIVE_FIELDS),
        }

    def update(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Validate, persist, and live-apply a partial settings update.

        Args:
            updates: A partial mapping of field name -> new value.

        Returns:
            The updated state (see :meth:`get_state`).

        Raises:
            ValueError: If a key is unknown or a value fails validation (the
                caller returns 400 and no state is changed).
            OSError: If the settings file cannot be written.
        """
        new = merge_settings(self._current, updates)
        save_settings(self._path, new)
        self._current = new
        self._apply_live(new)
        return self.get_state()

    def _apply_live(self, settings: Settings) -> None:
        """Apply the live-safe fields of ``settings`` to the running pipeline."""
        app_config.threshold = settings.detection_threshold
        app_config.included_classes = {
            name.lower() for name in settings.included_classes
        }
        apply_settings_to_context(settings, self._context)
        _set_log_level(settings.debug)

    def _needs_restart(self) -> bool:
        """Return True if a restart-only field differs from the booted value."""
        return any(
            getattr(self._current, name) != getattr(self._boot, name)
            for name in RESTART_FIELDS
        )
