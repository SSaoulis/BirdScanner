# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BirdScanner is a real-time bird detection and classification system designed for a Raspberry Pi with the Sony IMX500 AI Camera. The IMX500 runs object detection (SSD MobileNetV2) on-chip; detections are then passed to a ConvNeXt V2 Tiny ONNX model for bird species classification. The system must run on the Pi — `picamera2` and `libcamera` are not available on other platforms.

## Commands

### Run the app (on Raspberry Pi only)
```bash
cd src
python main.py --threshold 0.55 --object-duration-threshold 0.2 --multithread --debug
```

Key flags:
- `--multithread` — run classification on a background thread (prevents blocking the camera callback)
- `--object-duration-threshold <seconds>` — how long a track must be stable before classification fires (0 = legacy per-frame mode)
- `--debug` — enables `tracking` logger at DEBUG level
- `--preview` — shows the camera preview window

### Tests
```bash
# All tests
pytest tests/

# Single test
pytest tests/test_process_detections.py::test_update_tracks_for_frame_increments_stability

# With coverage
pytest --cov=src tests/
```

Tests import from `src.*` so run from the project root. Tests that require ONNX model files (e.g. `tests/test_classification.py`) skip automatically when the model is absent.

## Architecture

### Detection pipeline (frame loop)

```
IMX500 on-chip inference
  └─ parse_detections()         # extract Detection objects from inference tensor
       └─ StableDetectionTracker.update_frame()   # IoU-based multi-frame tracking
            └─ process_detections() [picam2 pre_callback]
                 └─ ClassificationManager.process()
                      └─ process_single_detection_with_stable_tracks()  # new path
                           └─ Classifier.classify()  # ConvNeXt V2 Tiny ONNX
                                └─ save full image + 200×200 thumbnail to IMAGE_DIR/{species}/
                                └─ DetectionWriter.write()  → SQLite detections table
```

### Key modules

**`src/object_detection.py`** — the bulk of the system logic:
- `Detection` — bounding box + category + confidence; box is set after coordinate conversion via `imx500.convert_inference_coords`
- `StableDetectionTracker` — IoU-based tracker; a detection must match across `min_stable_frames` consecutive frames before `should_run_bird_classification_for_detection` returns `True`; each track is classified at most once (`mark_classified`)
- `ClassificationManager` — wraps sync/async (threaded `Queue`) dispatch; in async mode items are dropped if the queue is full so the camera callback never blocks; accepts an optional `DetectionWriter` for DB persistence
- `process_single_detection_with_stable_tracks` — new gating path; `process_single_detection` is the legacy per-frame IoU-cache path (kept for reference)
- `IMAGE_DIR` — root directory for saved images, sourced from the `IMAGE_DIR` env var (defaults to `/home/stefan/Pictures/bird_detections`)

**`db/`** — SQLite persistence layer (Phase 1):
- `db/models.py` — `DetectionRecord` SQLModel ORM model (`detections` table)
- `db/database.py` — `make_engine()` / `init_db()` / `make_session_factory()`; DB path from `DB_PATH` env var
- `db/writer.py` — `DetectionWriter`: fire-and-forget background-thread writer; `write()` enqueues, `stop()` flushes and exits
- `db/migrations/001_initial.sql` — plain SQL migration (reference; `init_db()` is authoritative)

**`src/classification.py`** — ONNX classifier stack:
- `ONNXClassifier` — raw ONNX Runtime wrapper; expects `(N, C, H, W)` float32
- `Classifier` — adds preprocessing and class-index mapping; `classify()` returns `(species_str, confidence_float)`
- `build_preprocessing` — pure PIL+NumPy pipeline (resize → center crop → ImageNet normalize → NCHW)

**`src/main.py`** — entry point:
- Camera sensor crop is hardcoded to 900×900 anchored at `(4/13, 5/10)` of the 4056×3040 sensor (points the crop at the bird feeder)
- `vflip=True, hflip=True` transforms are applied (camera is mounted upside-down)
- Calls `update_detection_classifications_cache` each frame to keep the legacy temporal filter in sync alongside the new tracker

**`src/track_logging.py`** — `TrackingLogger` logs stable-track and track-deletion events to the `tracking` logger

### Model files (not in repo, must exist on the Pi)

| Purpose | Path (relative to `src/`) |
|---|---|
| Object detection (IMX500 firmware) | `/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` |
| Species classifier | `local/convnext_v2_tiny_int8.onnx` |
| Class-to-index mapping | `assets/convnext_v2_tiny.onnx_class_to_idx.json` |
| COCO labels | `assets/coco_labels.txt` |

### Bounding box format

All boxes throughout the codebase are `(x, y, w, h)` in ISP output pixel coordinates after `imx500.convert_inference_coords`. The `preprocess_roi` function expands the box to a square with 20% padding before passing to the classifier.

## Conventions

- All functions must have type hints and docstrings.
- `threading_logic.py` is a stale reference copy; the live `ClassificationManager` lives in `object_detection.py`.
- `examples/` contains the original unrefactored script; do not merge changes back into it.
- High-confidence classified bird images are written to `$IMAGE_DIR/{species}/` (env var, defaults to `/home/stefan/Pictures/bird_detections`). A 200×200 JPEG thumbnail is saved alongside each image with a `_thumb.jpg` suffix.
- `db/` tests use SQLAlchemy `StaticPool` to share an in-memory SQLite connection across threads.
