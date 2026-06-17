# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BirdScanner is a real-time bird detection and classification system designed for a Raspberry Pi with the Sony IMX500 AI Camera. The IMX500 runs object detection (YOLO11n) on-chip; detections are then passed to a ConvNeXt V2 Tiny ONNX model for bird species classification. The system must run on the Pi — `picamera2` and `libcamera` are not available on other platforms.

## Commands

### Run the app (on Raspberry Pi only)
```bash
cd src
python main.py
```

Runtime behaviour is configured in `src/config.py` (the former CLI args), not via
command-line flags. Edit the values on the module-level `config` instance:
- `multithread` — run classification on a background thread (prevents blocking the camera callback); defaults to `True`
- `object_duration_threshold` — seconds a track must be stable before classification fires (0 = legacy per-frame mode); defaults to `0.2`
- `debug` — enables `tracking` logger at DEBUG level; defaults to `True`
- `preview` — shows the camera preview window; defaults to `False`
- plus `model`, `fps`, `bbox_normalization`, `bbox_order`, `threshold`, `ignore_dash_labels`, `preserve_aspect_ratio`, `labels`, `print_intrinsics`

### Tests
```bash
# All tests
pytest tests/

# Single test
pytest tests/test_process_detections.py::test_update_tracks_for_frame_increments_stability

# With coverage
pytest --cov=src tests/
```

Tests import from `src.*` so run from the project root. Run them as `python -m pytest tests/` (module mode puts the project root on `sys.path` so the `src.*` / `db` imports resolve); plain `pytest tests/` fails collection. Tests that require ONNX model files (e.g. `tests/test_classification.py`) skip automatically when the model is absent.

### Type checking (mypy)

`mypy.ini` at the repo root configures mypy. `src/` modules import their siblings by
flat name (they run with `cwd=src`), while `tests/` import them via the `src.*` package
path — these two views cannot coexist in a single mypy process, so run mypy in **two
passes**:

```bash
mypy src backend db utils   # the source tree (flat imports)
mypy tests                  # the tests (src.* imports; mypy_path=src lets src's flat imports resolve)
```

`mypy.ini` sets `mypy_path = src` and silences missing-import noise for the Pi-only
native bindings (`libcamera`, `picamera2`) and the untyped third-party libs
(`onnxruntime`, `psutil`, `paramiko`), so mypy reports only genuine type errors in our
own code.

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
- `label_for_category` — bounds-checked label look-up; returns `None` when the class index is outside the label list. The IMX500 SSD model occasionally emits a spurious detection whose category index is out of range, so every `labels[category]` access goes through this helper. `draw_boxes` falls back to an `id:<n>` placeholder, and `process_detections` skips the detection entirely (logging a warning to the `tracking` logger) instead of crashing the camera callback with an `IndexError`

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
- `backend/main.py` — app factory; mounts the six routers; optionally serves `frontend/dist/` at `/` when the build exists via `SPAStaticFiles`, a `StaticFiles` subclass whose `get_response` falls back to `index.html` on a 404. The React app uses client-side routing (`BrowserRouter`), so deep links like `/history` have no file on disk; a plain `StaticFiles` mount returned `{"detail":"Not Found"}` on direct navigation/refresh. The fallback serves `index.html` so the SPA loads and renders the route client-side. API routes are unaffected (registered before the mount, so they take precedence)
- `backend/dependencies.py` — `get_session()`, `get_image_dir()`, and `get_reference_dir()` FastAPI dependency providers (reads `DB_PATH` / `IMAGE_DIR` / `SPECIES_REFERENCE_DIR` env vars); the engine is opened **read-only** (`make_engine(read_only=True)`) and the API never runs `init_db` — the detector owns schema creation, and the DB is mounted read-only. `get_reference_dir()` defaults to the repo-relative `assets/species_reference/` and may point at a non-existent dir — the reference API degrades gracefully when the manifest is absent
- `backend/routers/detections.py` — `GET /api/detections` (paginated + filtered by `species`, `from`, `to`, and `min_confidence` — a 0–1 floor applied as `confidence >= min_confidence`), `GET /api/detections/{id}`, and `DELETE /api/detections/{id}`. List results are ordered by `timestamp` **desc, then `id` desc** — the `id` tiebreaker keeps tied-timestamp rows in a deterministic order across paginated requests so offset-based pages don't overlap (overlap surfaced duplicate detections in the History tab). Since the API mounts the DB + images read-only, the delete is **proxied** to the detector's control server (`${DETECTOR_URL}/detections/{id}`, same channel as the camera snapshot) which owns the read-write data volume; relays 204 on success, 404 if the row is missing, 503 if the detector is unreachable
- `backend/routers/images.py` — `GET /api/images/{id}/thumbnail`, `GET /api/images/{id}/full`, `GET /api/images/download?ids=...` (chunked ZIP)
- `backend/routers/system.py` — `GET /api/system` (CPU/mem/disk/temp/uptime via psutil)
- `backend/routers/species.py` — `GET /api/species` (list with counts, sorted by count desc)
- `backend/routers/reference.py` — species **reference / comparison** data (prefix `/api/species`, tags `["reference"]`): `GET /api/species/{name}/reference` returns the cached Wikipedia/Wikidata info for a species (`common_name`, `scientific_name`, `summary`, `behaviour`, `wikipedia_url`, and an `images` list of `{url, attribution, license}`) — **404** when the species isn't in the manifest; `GET /api/species/{name}/reference/images/{index}` serves the locally-cached reference JPEG (path-traversal-guarded, 404 on out-of-range/missing). Reads the manifest from `get_reference_dir()` via a per-dir memoized loader; a missing/invalid `manifest.json` is treated as "no references" so every lookup 404s cleanly instead of crashing. The served image `url` points back at the images endpoint — the on-disk `path`/upstream `source_url` are never exposed. Powers the gallery/dashboard comparison panel
- `backend/routers/camera.py` — proxies the detector's snapshot + crop-control endpoints (`${DETECTOR_URL}`, default `http://detector:8000`) via httpx, since the API can't open the camera itself (the detector owns it exclusively + the data volume is read-only): `GET /api/camera/snapshot` → `/capture` (cropped feed), `GET /api/camera/snapshot/full` → `/capture/full` (full-sensor preview for the crop editor), `GET /api/camera/crop` → `/crop`, `POST /api/camera/crop` → `/crop` (relays the detector's status, e.g. 400 for a bad box). Returns 503 when the detector is unreachable

**`db/`** — SQLite persistence layer (Phase 1):
- `db/models.py` — `DetectionRecord` SQLModel ORM model (`detections` table)
- `db/database.py` — `make_engine()` / `init_db()` / `make_session_factory()`; DB path from `DB_PATH` env var. `make_engine(read_only=True)` opens the SQLite file via the `mode=ro` URI so it can be read off a read-only mount without attempting to create a journal file (used by the API)
- `db/writer.py` — `DetectionWriter`: fire-and-forget background-thread writer; `write()` enqueues, `stop()` flushes and exits
- `db/deleter.py` — `delete_detection(session_factory, image_dir, id)`: synchronously deletes a detection's DB row **and** its image + thumbnail files (best-effort file unlink — a missing file does not block the row delete); returns `False` when no row with that id exists. Runs in the detector (which owns the read-write data volume); the API proxies delete requests to it. Unlike the writer it is synchronous (it runs on the control server's HTTP thread, not the camera callback)
- `db/migrations/001_initial.sql` — plain SQL migration (reference; `init_db()` is authoritative)

**`tools/build_species_reference.py`** — offline species-reference data-bank builder (**not** part of the runtime; run on a dev machine with internet):
- Iterates every label in `src/assets/convnext_v2_tiny.onnx_class_to_idx.json` (the source of truth for "what species exist"; the `"Unknown"` sentinel is skipped) and fetches, per species: the **Wikipedia** REST summary (`summary` + lead image + canonical URL + Wikidata id), a best-effort "Behaviour"/"Habitat" section (`behaviour`) via the MediaWiki API, image attribution/license via `imageinfo`+`extmetadata`, and the **Wikidata** taxon name (P225) for `scientific_name`. Pure stdlib (`urllib`), descriptive `User-Agent`, polite throttling
- Writes `assets/species_reference/manifest.json` (schema: `{version, generated_at, source, species:{<label>:{common_name, scientific_name|null, summary, behaviour|null, wikipedia_url|null, images:[{path, source_url, attribution, license|null}]}}}`) + `coverage_report.json` (buckets: `missing`, `no_images`, `no_scientific_name`, `skipped`), and downloads each reference image to `assets/species_reference/images/<slug>/<n>.jpg` (`<slug>` = label lowercased, non-alphanumeric runs → single `-`)
- **Incremental / idempotent**: re-runs only fetch labels missing from the manifest (or whose image files are gone), so adding a new class = add the label to the classifier's `class_to_idx.json` + re-run. Flags: `--force` (refetch all), `--limit N` (only the first N labels needing a fetch — for a quick live smoke test), `--throttle SECONDS`
- `assets/species_reference/overrides.json` — hand-maintained `{<label>:{wikipedia_title?, scientific_name?, skip?}}`; consulted first, so it fixes name→article mismatches (e.g. apostrophe-stripped labels like `Audouins gull` → `Audouin's gull`) and forces `skip` for `"Unknown"`. This file is **source** (curate + commit it); the generated `manifest.json`/`coverage_report.json` and the downloaded `images/` are **build artifacts** shipped/mounted out-of-band (like the ONNX model — a full build is ~hundreds of MB of JPEGs), and are `.gitignore`d (a `!assets/species_reference/overrides.json` negation keeps the curated overrides tracked despite the repo-wide `*.json` rule). The API finds the data via `SPECIES_REFERENCE_DIR` (default `assets/species_reference/`). **In the Docker stack** the bank is **not** baked into the api image (`Dockerfile.api` copies only `db/`, `backend/`, `frontend/dist`); instead it lives on the data volume at `/data/species_reference` with `SPECIES_REFERENCE_DIR=/data/species_reference` (set in `.env.example`). Build it on a dev machine and copy `manifest.json` + `images/` into `<data-volume>/species_reference/` on the host (e.g. `/mnt/ssd/birdfinder-data/species_reference/`); the api mounts the data volume read-only and reads it there. If this dir is missing/empty, **every** reference lookup 404s and the Lightbox panel shows "No reference available" for all species — that is the symptom of an un-deployed bank, not a code bug

**`src/classification.py`** — ONNX classifier stack:
- `ONNXClassifier` — raw ONNX Runtime wrapper; expects `(N, C, H, W)` float32
- `Classifier` — adds preprocessing and class-index mapping; `classify()` returns `(species_str, confidence_float)`
- `build_preprocessing` — pure PIL+NumPy pipeline (resize → center crop → ImageNet normalize → NCHW)

**`src/config.py`** — application configuration:
- `Config` dataclass + module-level `config` instance holding every runtime setting that used to be a CLI arg (`model`, `fps`, `bbox_normalization`, `bbox_order`, `threshold`, `ignore_dash_labels`, `preserve_aspect_ratio`, `labels`, `print_intrinsics`, `multithread`, `object_duration_threshold`, `debug`, `preview`); edit values here instead of passing flags. `main.py` imports it as `app_config` (to avoid clashing with the local `config` returned by `picam2.create_preview_configuration`)

**`src/main.py`** — entry point:
- Reads all runtime settings from `config.config` (imported as `app_config`); no `argparse`. The intrinsics-override loop iterates `vars(app_config)`, so optional intrinsic fields left as `None` do not clobber the network intrinsics defaults
- The detection crop region is **variable and set from the UI** (no longer hardcoded). On startup it loads the persisted region via `load_crop_region(crop_config_path(), default_crop_region(...))` (default = the historical 900×900 at `(4/13, 5/10)` of the 4056×3040 sensor, aimed at the feeder), sizes the `main` stream to the crop's aspect ratio (`main_stream_size_for_crop`), and builds the camera config through a `build_camera_config(main_size, scaler_crop)` closure that centralises every picamera2 knob so `CropController` can rebuild it on a reconfigure
- A `CropController` (see `src/crop_controller.py`) owns all live crop changes and exposes a `camera_lock`; the main capture loop holds that lock across `capture_metadata()` + `parse_detections()` so a UI-triggered reconfigure (which may `stop()`/`start()` the camera) never races an in-flight capture
- `vflip=True, hflip=True` transforms are applied (camera is mounted upside-down)
- Calls `update_detection_classifications_cache` each frame to keep the legacy temporal filter in sync alongside the new tracker
- Creates the SQLite engine and runs `init_db()` on startup **before camera init**, so the schema always exists; the detector owns all DB writes, so this lets the read-only API serve an empty gallery even when the camera never comes up. The same `engine` is reused to wire a `DetectionWriter` into the `ClassificationManager` so every high-confidence classification is persisted; the writer is flushed via `detection_writer.stop()` on `KeyboardInterrupt`.
- `wait_for_camera()` wraps `IMX500(...)` in a retry-with-backoff loop (30 s default): when the camera dev-node is missing it logs a concise warning and retries instead of crashing, so the detector never crash-loops and auto-recovers when the camera reappears

**`frontend/`** — React + Vite + Tailwind dashboard (Phase 3 & 4):
- `frontend/src/api.ts` — typed fetch wrappers for all `/api/*` endpoints (including `detections.delete(id)` → `DELETE /api/detections/{id}`); exports `Detection`, `SystemStatus`, `SpeciesSummary`, `CropState`, `NormalizedBox`, `SpeciesReference`, `SpeciesReferenceImage` interfaces plus `timeAgo` and `formatUptime` helpers. `api.camera` exposes `snapshotUrl`, `fullSnapshotUrl`, `getCrop`, and `setCrop` (POSTs a normalized box or `{reset:true}`); `api.species.reference(name)` fetches a species' reference data. `apiFetch` throws an `ApiError` carrying the HTTP `status` so callers can distinguish a 404 (e.g. "no reference yet") from a real error
- `frontend/src/App.tsx` — root component; sets up `react-router-dom` `BrowserRouter` with routes `/` → Dashboard, `/history` → History, and `/camera` → Camera; renders a top-level nav bar
- `frontend/src/components/SystemMonitor.tsx` — polls `/api/system` every 5 s; renders animated gauge bars (green/yellow/red) for CPU, memory, disk, temp, and uptime
- `frontend/src/components/DetectionCard.tsx` — loads thumbnail from `/api/images/{id}/thumbnail`; shows species, confidence %, and time-ago label; supports optional `onSelect`/`selected` props for bulk-select mode and `onOpenLightbox` prop to trigger the lightbox
- `frontend/src/pages/Dashboard.tsx` — composes `SystemMonitor` + a horizontal-scroll strip of the last 10 `DetectionCard`s; a min-confidence slider (0–100%) refetches the strip with the `min_confidence` query param (0 = show all). The slider tracks a live display value and only commits to the fetch on release (mouse up / touch end / key up), so dragging doesn't fire a request per intermediate value. Cards open the `Lightbox` species comparison panel on click (the Dashboard owns its own `lightboxIndex` state, reset when the strip reloads or a detection is deleted)
- `frontend/src/pages/History.tsx` — full-page history view; filter bar (species dropdown + from/to date pickers + min-confidence slider mapped to the `min_confidence` query param, 0 = show all; the slider only commits to the fetch on release so dragging doesn't refetch per intermediate value), tab switcher (Timeline | Gallery), infinite-scroll pagination (20/page via `IntersectionObserver`); owns all filter/pagination/lightbox/selection state and passes it down to sub-views. Filtering is done server-side so pagination stays correct. When appending a page it **dedupes by `id`** (keeps only rows not already held) before merging: the detector writes new detections live, so a row inserted between page fetches shifts the offset window and makes the next page re-return rows already shown — without the dedupe those rows render twice in both Timeline and Gallery (which share this one list). The offset cursor still advances by the raw page length. `removeDetections(ids)` reaps already-deleted rows from local state (list + selection + pagination offset) and closes the lightbox — the actual API deletes happen in `Lightbox`/`FileDownloader`, this only keeps the UI in sync
- `frontend/src/pages/Camera.tsx` — Camera tab with two sub-tabs. **Test**: a "Test Camera" button fetches `/api/camera/snapshot` as a blob (so HTTP errors surface as a message, not a broken image), cache-busts per click, and displays the returned frame; revokes object URLs on replace/unmount. **Detection region**: renders `CropEditor`
- `frontend/src/components/CropEditor.tsx` — interactive detection-region editor; loads a full-sensor preview (`/api/camera/snapshot/full`) rendered at the true sensor aspect ratio (`object-fit: fill` into an aspect-ratio container) and overlays a draggable/resizable box (corner handles, `pointer` events) tracked in normalized `[0,1]` coords over the displayed preview — exactly the space `POST /api/camera/crop` expects. "Apply region" persists + applies it live; "Reset to default" restores the feeder crop; "Reload preview" refetches
- `frontend/src/components/Timeline.tsx` — chronological paginated list of `DetectionCard`s with an `IntersectionObserver` sentinel for infinite scroll; opens lightbox on thumbnail click; full-res images are never loaded until the lightbox is opened
- `frontend/src/components/Gallery.tsx` — uniform thumbnail grid with checkbox-based multi-select (checkbox overlay + ring); `IntersectionObserver` for infinite scroll; integrates `FileDownloader` toolbar; opens lightbox on thumbnail click
- `frontend/src/components/Lightbox.tsx` — full-screen **species comparison panel** opened on thumbnail click (Gallery, Timeline, and Dashboard) — an in-app overlay, never a new page. Left pane: the captured full-res image with the species/confidence/time-ago caption, a download link, and a **Delete** action (confirm → `api.detections.delete` → `onDelete(id)`). Right pane (`ReferencePane`): fetches `api.species.reference(detection.species)` and shows the reference image(s) (prominent image + thumbnail strip when there are several), scientific name, summary, behaviour, per-image attribution/license, and a Wikipedia link (the only external nav). Refetches whenever prev/next changes the species (stale-response guard); graceful states for loading, 404 ("No reference available for this species yet"), and errors. `Timeline`/`Gallery`/`History`/`Dashboard` pass `onDelete` through. Keeps Esc/←/→ keyboard nav, prev/next arrows, body-scroll lock, and backdrop-click-to-close. Responsive: side-by-side on wide screens, stacked on narrow
- `frontend/src/components/FileDownloader.tsx` — manages bulk-select toolbar (select all / clear / count); streams ZIP download via `fetch()` + `ReadableStream`; tracks progress from `Content-Length` header; triggers browser download via `URL.createObjectURL`. Also renders a bulk **Delete** button: confirms, deletes the selected ids one-by-one via `api.detections.delete`, and reports the successfully-deleted ids to `onDeleteSelected` (partial failures still remove what succeeded)
- Build: `npm run build` (from `frontend/`) outputs to `frontend/dist/`; served by FastAPI at `/` via `StaticFiles`
- Dev: `npm run dev` proxies `/api/*` to `http://localhost:8080`
- `Dockerfile.api` — multi-stage image: Node 20 builds the frontend, Python 3.11 runs the API; exposes port 8080

**`src/camera_server.py`** — detector control HTTP server (camera snapshots + crop control + detection deletion):
- `encode_jpeg(frame)` / `capture_jpeg(picam2)` — encode an RGB frame (RGB→BGR for OpenCV; the stream is `BGR888`, see "Colour channel order") / capture one from the `main` stream and encode it
- `start_camera_server(picam2, port, crop_controller=None, delete_detection=None)` — runs a stdlib `ThreadingHTTPServer` on a background daemon thread. Routes: `GET /capture` (fresh JPEG of the cropped feed); when a `crop_controller` is supplied, `GET /capture/full` (full-sensor preview), `GET /crop` (current region as JSON), `POST /crop` (apply `{nx,ny,nw,nh}` or `{reset:true}`); and when a `delete_detection` callback is supplied, `DELETE /detections/{id}` (returns 204/404/500). With no controller the crop routes 404 (legacy snapshot-only mode); with no delete callback the delete route 404s. `camera_server_port()` reads `CAMERA_SERVER_PORT` (default 8000)
- Wired into `main.py` after `picam2.start(...)` so the read-only API can surface a live "Test Camera" image, the crop editor, **and** delete detections even though the detector owns the camera + read-write data volume exclusively; the delete callback is bound to the same engine the `DetectionWriter` uses (see `db/deleter.py`). Shut down on `KeyboardInterrupt`. NB: because the server starts only after the camera initialises, deletes (like snapshots/crop) are unavailable while the camera is down

**`src/crop.py`** — pure, camera-independent crop domain (fully unit-tested in `tests/test_crop.py`):
- `CropRegion` — `(x, y, w, h)` in **unflipped raw sensor pixels**; `.clamped()` enforces `MIN_CROP_PX` and sensor bounds (`SENSOR_W=4056`, `SENSOR_H=3040`)
- `default_crop_region()` — the historical 900×900 feeder crop
- `normalized_to_sensor()` / `sensor_to_normalized()` — convert between a UI box (`nx,ny,nw,nh` fractions over the displayed preview) and sensor coords via a **direct per-axis scale, no rotation**: libcamera applies `ScalerCrop` in the same orientation as the (vflip+hflip) transformed preview, so the displayed box maps straight to the matching sensor region (verified empirically — inverting the rotation cropped the diagonally-opposite corner)
- `main_stream_size_for_crop()` — the `main` ISP output size matching a crop's aspect ratio (longer edge `DEFAULT_LONG_SIDE=640`, even-aligned), so a non-square crop is not stretched
- `load_crop_region()` / `save_crop_region()` (atomic) — JSON persistence at `crop_config_path()` (env `CROP_CONFIG_PATH`, default `/data/crop.json`); any read error falls back to the default so a corrupt file never blocks startup

**`src/crop_controller.py`** — `CropController` applies + persists the crop on the running camera, serialising all camera access behind `camera_lock` (an `RLock` shared with `main.py`'s capture loop):
- `set_from_normalized()` / `reset_to_default()` → `_apply()`: a **pan/zoom at the same aspect ratio** applies live via `set_controls({"ScalerCrop": ...})`; an **aspect-ratio change** triggers a `stop()`→`configure()`→`start()` reconfigure (using the `config_factory` from `main.py`) to resize the `main` stream
- `capture_full_preview_array()` — briefly widens `ScalerCrop` to the full sensor under the lock, pulls a settled frame (`_capture_settled` waits for the control to take effect), then restores the previous crop; the momentary widening is a deliberate, brief glitch in the live feed while configuring
- `get_state()` — returns the crop as sensor pixels + normalized box + sensor dimensions (for the UI)

**`src/track_logging.py`** — `TrackingLogger` logs stable-track and track-deletion events to the `tracking` logger

### Model files (not in repo, must exist on the Pi)

| Purpose | Path (relative to `src/`) |
|---|---|
| Object detection (IMX500 firmware) | `/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk` (YOLO11n; emits normalized, `xy`-ordered boxes — see `src/config.py`) |
| Species classifier | `local/convnext_v2_tiny_int8.onnx` |
| Class-to-index mapping | `assets/convnext_v2_tiny.onnx_class_to_idx.json` |
| COCO labels | `assets/coco_labels.txt` |

### Bounding box format

All boxes throughout the codebase are `(x, y, w, h)` in ISP output pixel coordinates after `imx500.convert_inference_coords`. The `preprocess_roi` function expands the box to a square with 20% padding before passing to the classifier.

### Colour channel order (RGB vs BGR)

The `main` ISP stream is configured with `"format": "BGR888"` — **not** `"RGB888"`. picamera2's `888` format names are byte-reversed relative to the numpy array they produce: `"BGR888"` yields an `[R, G, B]`-ordered array, while `"RGB888"` yields `[B, G, R]`. Everything downstream assumes **RGB**: the ConvNeXt classifier (ImageNet-RGB normalisation in `build_preprocessing`), `save_thumbnail` (`PIL.Image.fromarray`), and the `cv2.cvtColor(..., RGB2BGR)` writes in `classification_pipeline.py` / `camera_server.py`. Requesting `"RGB888"` therefore swaps red↔blue in every saved image **and** feeds the classifier mis-ordered channels, producing garbage species predictions. If colours ever invert again, check this format string first.

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
- The `data` Docker volume is the single source of truth: the `detector` service writes images, the SQLite DB, and the crop-region JSON (`CROP_CONFIG_PATH`, default `/data/crop.json`); the `api` service mounts it read-only. The crop config flows the other way too — the API never writes the file; it `POST`s crop changes to the detector's HTTP server, which applies them live and persists the JSON.
- The **species-reference bank** also rides on the data volume at `/data/species_reference` (`SPECIES_REFERENCE_DIR`). It is built offline (`tools/build_species_reference.py`) and **not** baked into the api image — copy `manifest.json` + `images/` into `<data-volume>/species_reference/` on the host. Neither service writes it; the api just reads it (read-only mount). When absent, the Lightbox reference panel 404s for every species ("No reference available").
- `privileged: true` is scoped to `detector` only (required for IMX500 camera device access).
- The `detector` `expose`s port 8000 (the control server from `src/camera_server.py`) on the internal compose network only — it is **not** published to the host. The `api` reaches it via `DETECTOR_URL` (default `http://detector:8000`); the detector binds the port from `CAMERA_SERVER_PORT` (default 8000). Both vars live in `.env`/`.env.example` and the defaults work as-is — only change them together. This powers the Camera tab's "Test Camera" button, the detection-region editor (`/crop`, `/capture/full`), **and** detection deletes (`DELETE /api/detections/{id}` proxies to `${DETECTOR_URL}/detections/{id}`).
- The `detector` image (`Dockerfile.detector`) is based on `dtcooper/raspberrypi-os:bookworm` so the system `python3` can import the apt-installed `python3-picamera2` / `python3-libcamera` bindings (these are built natively against the Pi's libcamera and are **not** on PyPI). `numpy`, `opencv` and `pillow` are installed via apt (`python3-numpy` / `python3-opencv` / `python3-pil`); the apt `numpy` is `1.24.2`, which is what the apt-built picamera2/simplejpeg/opencv stack is compiled against. `requirements.detector.txt` therefore pins `numpy==1.24.2` (so pip does not pull numpy 2.x into `/usr/local` and shadow the apt copy — that crashes the detector with `numpy.dtype size changed`) and `onnxruntime==1.23.2` (the version known to run against numpy 1.24.2 on this Pi). These plus `sqlmodel` are pip-installed into the system interpreter with `--break-system-packages`. The IMX500 firmware + `.rpk` network models (`/usr/share/imx500-models/...`) come from the `imx500-all` apt package baked into the image. A plain `python:3.x` base will crash the detector with `ModuleNotFoundError: No module named 'libcamera'`.
- The `detector` service additionally mounts `/run/udev:ro` (libcamera enumerates cameras via udev), the `/dev/dma_heap` device (picamera2 buffer allocation), `/sys/kernel/debug` (debugfs), and `/usr/share/imx500-models:ro` — all required for the camera to initialise inside the container. The `imx500-models` mount exposes the **host's** `.rpk` network firmware to the container: the image's `imx500-all` apt package does not ship the `imx500_network_yolo11n_pp.rpk` model the detector loads (see `src/config.py`), so without the mount the detector logs `Camera not available (Firmware file ...yolo11n_pp.rpk does not exist.)` and retry-loops forever. Mounting the host directory (which has the model, since the bare detector runs there) keeps the container in sync with the host's apt package without baking the binary into the image. The debugfs mount is needed because `IMX500.__init__` opens `/sys/kernel/debug/imx500-fw:<id>/fw_progress` to track on-chip firmware upload; Docker does not expose debugfs to containers (even privileged ones) by default, so without this mount the detector crashes with `FileNotFoundError: ... /sys/kernel/debug/imx500-fw:11-001a/fw_progress`.
- `main.py` configures the camera with `buffer_count=6` (not the bare-Pi example's `12`). Inside the container the kernel CMA / dma-heap pool is host-global (Docker memory limits don't apply to it) and is shared with the IMX500 firmware upload and the always-allocated 2028×1520 raw sensor stream (~3 MB/buffer, the dominant consumer — it's fixed at the sensor's smallest mode and allocated by the Pi 5 PiSP pipeline regardless of config, so it can't be shrunk). With 12 buffers this exhausted CMA and crashed `picam2.start()` with `OSError: [Errno 12] Cannot allocate memory` at the dma-heap `alloc` ioctl. 6 buffers halves DMA pressure while keeping jitter margin; the `main` stream is sized so its longer edge is 640 (square for a square crop, smaller on the short side for a non-square crop — see `main_stream_size_for_crop`), so it never exceeds the old 640×640 footprint. The ConvNeXt classifier crops bird ROIs from `main` (the on-chip detector runs at a fixed 320×320 baked into the `.rpk`, independent of the streams). If 6 still ENOMEMs, increase the host CMA pool in `/boot/firmware/config.txt` rather than dropping `main` resolution.
- Model files (`src/local/convnext_v2_tiny_int8.onnx`) must be present on the Pi; they are not included in the image.
- `main.py` runs with `cwd=/app` (the `CMD` is `python3 src/main.py` from `WORKDIR /app`) and loads its data files via paths relative to that cwd — `assets/coco_labels.txt` and `local/convnext_v2_tiny_int8.onnx`. The Dockerfile therefore copies them to **top-level** `/app/assets/` and `/app/local/` respectively (NOT `/app/src/...`); copying the model under `/app/src/local/` makes onnxruntime fail at startup with `NO_SUCHFILE: Load model from local/convnext_v2_tiny_int8.onnx failed`. Only the single `convnext_v2_tiny_int8.onnx` is copied — the rest of `src/local/` is training/conversion scratch.
- If the camera is unavailable (dev-node missing — unplugged, mis-seated, or device access not granted), the `detector` does **not** crash-loop: `wait_for_camera()` logs `Camera not available (...). Retrying in 30s...` and retries until the camera appears. The `api` service is independent and continues serving the site; because the detector initialises the DB schema on startup, the UI loads (showing existing images, or an empty gallery on a fresh volume).
