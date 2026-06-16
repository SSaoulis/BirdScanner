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

The detection pipeline is split across four focused modules (the previously
monolithic `object_detection.py` was refactored along these seams):

**`src/object_detection.py`** — core object detection only:
- `Detection` — bounding box + category + confidence; box is set after coordinate conversion via `imx500.convert_inference_coords`
- `parse_detections` — extracts `Detection` objects from the IMX500 inference tensor; `get_labels` filters the intrinsics label list
- `last_detections` — module-global fallback returned when a frame yields no inference output

**`src/detection_utils.py`** — stateless geometry/drawing helpers shared across the pipeline:
- `iou` — Intersection-over-Union for two `(x, y, w, h)` boxes
- `preprocess_roi` — expands a box to a padded square and crops the ROI
- `draw_boxes` — annotates a frame with boxes + labels + optional classification result
- `save_thumbnail` — writes a 200×200 JPEG thumbnail

**`src/tracking.py`** — multi-frame stability tracking:
- `StableDetectionTracker` — IoU-based tracker; a detection must match across `min_stable_frames` consecutive frames before `should_run_bird_classification_for_detection` returns `True`; each track is classified at most once (`mark_classified`)
- `StableTrack`, `match_detection_to_track`, `update_tracks_for_frame`, `should_classify_track` — the underlying pure-function tracking primitives (directly unit-tested in `tests/test_process_detections.py`)
- `stable_detection_tracker` — module-global default tracker instance

**`src/classification_pipeline.py`** — classification orchestration + persistence:
- `ClassificationManager` — wraps sync/async (threaded `Queue`) dispatch; in async mode items are dropped if the queue is full so the camera callback never blocks; accepts an optional `DetectionWriter` for DB persistence
- `process_detections` — picam2 `pre_callback` entry point; draws boxes and queues bird detections
- `process_single_detection_with_stable_tracks` — new gating path; `process_single_detection` is the legacy per-frame IoU-cache path (kept for reference)
- `setup_classifier`, `run_bird_classification`, `update_detection_classifications_cache`, `classification_results`
- `IMAGE_DIR` — root directory for saved images, sourced from the `IMAGE_DIR` env var (defaults to `/home/stefan/Pictures/bird_detections`)

**`backend/`** — FastAPI REST API (Phase 2):
- `backend/main.py` — app factory; mounts the four routers; optionally serves `frontend/dist/` at `/` when the build exists
- `backend/dependencies.py` — `get_session()` and `get_image_dir()` FastAPI dependency providers (reads `DB_PATH` / `IMAGE_DIR` env vars); the engine is opened **read-only** (`make_engine(read_only=True)`) and the API never runs `init_db` — the detector owns schema creation, and the DB is mounted read-only
- `backend/routers/detections.py` — `GET /api/detections` (paginated + filtered) and `GET /api/detections/{id}`
- `backend/routers/images.py` — `GET /api/images/{id}/thumbnail`, `GET /api/images/{id}/full`, `GET /api/images/download?ids=...` (chunked ZIP)
- `backend/routers/system.py` — `GET /api/system` (CPU/mem/disk/temp/uptime via psutil)
- `backend/routers/species.py` — `GET /api/species` (list with counts, sorted by count desc)

**`db/`** — SQLite persistence layer (Phase 1):
- `db/models.py` — `DetectionRecord` SQLModel ORM model (`detections` table)
- `db/database.py` — `make_engine()` / `init_db()` / `make_session_factory()`; DB path from `DB_PATH` env var. `make_engine(read_only=True)` opens the SQLite file via the `mode=ro` URI so it can be read off a read-only mount without attempting to create a journal file (used by the API)
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
- Creates the SQLite engine and runs `init_db()` on startup **before camera init**, so the schema always exists; the detector owns all DB writes, so this lets the read-only API serve an empty gallery even when the camera never comes up. The same `engine` is reused to wire a `DetectionWriter` into the `ClassificationManager` so every high-confidence classification is persisted; the writer is flushed via `detection_writer.stop()` on `KeyboardInterrupt`.
- `wait_for_camera()` wraps `IMX500(...)` in a retry-with-backoff loop (30 s default): when the camera dev-node is missing it logs a concise warning and retries instead of crashing, so the detector never crash-loops and auto-recovers when the camera reappears

**`frontend/`** — React + Vite + Tailwind dashboard (Phase 3 & 4):
- `frontend/src/api.ts` — typed fetch wrappers for all `/api/*` endpoints; exports `Detection`, `SystemStatus`, `SpeciesSummary` interfaces plus `timeAgo` and `formatUptime` helpers
- `frontend/src/App.tsx` — root component; sets up `react-router-dom` `BrowserRouter` with routes `/` → Dashboard and `/history` → History; renders a top-level nav bar
- `frontend/src/components/SystemMonitor.tsx` — polls `/api/system` every 5 s; renders animated gauge bars (green/yellow/red) for CPU, memory, disk, temp, and uptime
- `frontend/src/components/DetectionCard.tsx` — loads thumbnail from `/api/images/{id}/thumbnail`; shows species, confidence %, and time-ago label; supports optional `onSelect`/`selected` props for bulk-select mode and `onOpenLightbox` prop to trigger the lightbox
- `frontend/src/pages/Dashboard.tsx` — composes `SystemMonitor` + a horizontal-scroll strip of the last 10 `DetectionCard`s
- `frontend/src/pages/History.tsx` — full-page history view; filter bar (species dropdown + from/to date pickers), tab switcher (Timeline | Gallery), infinite-scroll pagination (20/page via `IntersectionObserver`); owns all filter/pagination/lightbox/selection state and passes it down to sub-views
- `frontend/src/components/Timeline.tsx` — chronological paginated list of `DetectionCard`s with an `IntersectionObserver` sentinel for infinite scroll; opens lightbox on thumbnail click; full-res images are never loaded until the lightbox is opened
- `frontend/src/components/Gallery.tsx` — uniform thumbnail grid with checkbox-based multi-select (checkbox overlay + ring); `IntersectionObserver` for infinite scroll; integrates `FileDownloader` toolbar; opens lightbox on thumbnail click
- `frontend/src/components/Lightbox.tsx` — full-screen modal showing the full-res image on open; Esc/arrow-key keyboard navigation; prev/next arrows; caption bar with species, confidence, and time-ago; download link
- `frontend/src/components/FileDownloader.tsx` — manages bulk-select toolbar (select all / clear / count); streams ZIP download via `fetch()` + `ReadableStream`; tracks progress from `Content-Length` header; triggers browser download via `URL.createObjectURL`
- Build: `npm run build` (from `frontend/`) outputs to `frontend/dist/`; served by FastAPI at `/` via `StaticFiles`
- Dev: `npm run dev` proxies `/api/*` to `http://localhost:8080`
- `Dockerfile.api` — multi-stage image: Node 20 builds the frontend, Python 3.11 runs the API; exposes port 8080

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
- `threading_logic.py` is a stale reference copy; the live `ClassificationManager` lives in `classification_pipeline.py`.
- `examples/` contains the original unrefactored script; do not merge changes back into it.
- High-confidence classified bird images are written to `$IMAGE_DIR/{species}/` (env var, defaults to `/home/stefan/Pictures/bird_detections`). A 200×200 JPEG thumbnail is saved alongside each image with a `_thumb.jpg` suffix.
- `db/` tests use SQLAlchemy `StaticPool` to share an in-memory SQLite connection across threads.
- `backend/` tests (``tests/test_backend.py``) override FastAPI dependencies via ``app.dependency_overrides`` so no real DB or filesystem is needed.

## Deployment

The system is packaged as a Docker Compose stack (`docker-compose.yml`) with two services: `detector` (the ML pipeline) and `api` (FastAPI + React frontend).

### First-time setup
```bash
cp .env.example .env
# Edit .env if you need non-default paths or ports
docker compose up --build
```

### Normal operation
```bash
# Start all services (detached)
docker compose up -d

# Stop without losing data (named volume persists)
docker compose down && docker compose up -d

# Tail detector logs
docker compose logs -f detector
```

### Accessing the UI
- On Pi with mDNS (avahi): `http://birdpi.local:8080`
- Direct IP: `http://<pi-ip>:8080`

### Notes
- `.env` must be created from `.env.example` before the first run (it is git-ignored).
- The `data` Docker volume is the single source of truth: the `detector` service writes images and the SQLite DB; the `api` service mounts it read-only.
- `privileged: true` is scoped to `detector` only (required for IMX500 camera device access).
- The `detector` image (`Dockerfile.detector`) is based on `dtcooper/raspberrypi-os:bookworm` so the system `python3` can import the apt-installed `python3-picamera2` / `python3-libcamera` bindings (these are built natively against the Pi's libcamera and are **not** on PyPI). `numpy`, `opencv` and `pillow` are installed via apt (`python3-numpy` / `python3-opencv` / `python3-pil`); the apt `numpy` is `1.24.2`, which is what the apt-built picamera2/simplejpeg/opencv stack is compiled against. `requirements.detector.txt` therefore pins `numpy==1.24.2` (so pip does not pull numpy 2.x into `/usr/local` and shadow the apt copy — that crashes the detector with `numpy.dtype size changed`) and `onnxruntime==1.23.2` (the version known to run against numpy 1.24.2 on this Pi). These plus `sqlmodel` are pip-installed into the system interpreter with `--break-system-packages`. The IMX500 firmware + `.rpk` network models (`/usr/share/imx500-models/...`) come from the `imx500-all` apt package baked into the image. A plain `python:3.x` base will crash the detector with `ModuleNotFoundError: No module named 'libcamera'`.
- The `detector` service additionally mounts `/run/udev:ro` (libcamera enumerates cameras via udev) and the `/dev/dma_heap` device (picamera2 buffer allocation) — both required for the camera to initialise inside the container.
- Model files (`src/local/convnext_v2_tiny_int8.onnx`) must be present on the Pi; they are not included in the image.
- If the camera is unavailable (dev-node missing — unplugged, mis-seated, or device access not granted), the `detector` does **not** crash-loop: `wait_for_camera()` logs `Camera not available (...). Retrying in 30s...` and retries until the camera appears. The `api` service is independent and continues serving the site; because the detector initialises the DB schema on startup, the UI loads (showing existing images, or an empty gallery on a fresh volume).
