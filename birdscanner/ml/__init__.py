"""Platform-independent detection and classification inference.

These modules carry no camera/hardware dependency, so they import and unit-test off the
Pi. They must never import from :mod:`birdscanner.detector` or :mod:`birdscanner.api`.
"""
