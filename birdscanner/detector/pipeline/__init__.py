"""Classification-pipeline wiring: gating, tracking logs, and the geomodel prior.

Assembles the stable-track gating machinery and :class:`ClassificationManager`
(:mod:`~birdscanner.detector.pipeline.gating`), the ``tracking`` logger
(:mod:`~birdscanner.detector.pipeline.track_logging`), and the startup geomodel-prior
refresh (:mod:`~birdscanner.detector.pipeline.geo_priors`). Imports ``ml`` / ``db`` but
never ``picamera2``, so it stays importable off the Pi.
"""
