"""Dev/test-only tooling that is never part of the runtime services.

Modules here are used for local development and the test suite only — they are
excluded from the Docker images (like ``tools/``) and must not be imported by
``birdscanner`` runtime code.  Currently this holds the off-Pi IMX500 camera
emulator (:mod:`dev.emulation`, driven by :mod:`dev.run_emulated`).
"""
