"""BirdScanner: real-time bird detection and classification for the Raspberry Pi.

The package is organised into four layered subpackages:

- :mod:`birdscanner.detector` — Pi-only camera + hardware control and the entry point.
- :mod:`birdscanner.ml` — platform-independent detection/classification inference.
- :mod:`birdscanner.api` — the FastAPI REST API serving the React dashboard.
- :mod:`birdscanner.db` — SQLite persistence.

Import direction is one-way: ``detector -> ml -> db`` and ``api -> db``.
"""
