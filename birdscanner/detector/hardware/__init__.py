"""Pi-only camera bring-up, streams, crop, recording, and the control server.

Modules here depend on :mod:`picamera2` / :mod:`libcamera` (Raspberry Pi only) or on
the camera's ISP output frames: camera bring-up (:mod:`~birdscanner.detector.hardware.camera`),
the crop domain + live controller, the full-FOV raw-frame extractor, the short-clip
recorder, and the detector control HTTP server.
"""
