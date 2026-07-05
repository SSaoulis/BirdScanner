# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BirdScanner is a real-time bird detection and classification system designed for a Raspberry Pi with the Sony IMX500 AI Camera. The IMX500 runs object detection (YOLO11n) on-chip; detections are then passed to a ConvNeXt V2 Tiny ONNX model for bird species classification. The system must run on the Pi — `picamera2` and `libcamera` are not available on other platforms.

## How to develop

The development cycle is based on a worktree approach. Therefore there are a fixed set of rules to apply to develop: 

- Start by creating a worktree from main.
- ALWAYS pull latest changes from main onto the worktree before any files are read/analysed.
- Once a change is made, ALWAYS update the CLAUDE.md, and make a PR.

Use the .venv in the project root - this contains the linting/formatting tools and all required packages installed. 

## Commands

### Run the app (on Raspberry Pi only)
```bash
# from the repo root
python -m birdscanner.detector.main
```

Runtime behaviour is configured in `birdscanner/detector/config.py` (the former CLI args), not via
command-line flags. Edit the values on the module-level `config` instance:
- `multithread` — run classification on a background thread (prevents blocking the camera callback); defaults to `True`
- `object_duration_threshold` — seconds a track must be stable before classification fires (0 = legacy per-frame mode); defaults to `0.2`
- `debug` — enables `tracking` logger at DEBUG level; defaults to `True`
- `preview` — shows the camera preview window; defaults to `False`
- `save_video` — save a short mp4 clip per detection (best-frame still is always saved); defaults to `True`. `video_pre_roll_seconds` / `video_post_roll_seconds` size the clip (defaults `3.0` / `4.0`)
- plus `model`, `fps`, `bbox_normalization`, `bbox_order`, `threshold`, `ignore_dash_labels`, `preserve_aspect_ratio`, `labels`, `print_intrinsics`

### Tests
```bash
# All tests
pytest tests/

# Single test
pytest tests/ml/test_tracking.py::test_update_tracks_for_frame_increments_stability

# With coverage
pytest --cov=birdscanner tests/
```

The whole codebase is one package (`birdscanner`), so everything imports it via the
absolute `birdscanner.*` path. `pyproject.toml` sets `[tool.pytest.ini_options] pythonpath = ["."]`, so a plain `pytest tests/` from the repo root resolves the package with no editable install or module-mode invocation. Tests that require ONNX model files (e.g. `tests/ml/test_classification.py`) skip automatically when the model is absent.

**Test layout** — `tests/` mirrors the `birdscanner/` package: `tests/ml/`,
`tests/db/`, `tests/api/` (with `tests/api/routers/` per router + `tests/api/test_main.py`
for the app factory / SPA mount), `tests/detector/`, and `tests/tools/`. Each directory
is a package (`__init__.py`), so duplicate basenames across layers are safe. Shared
fixtures live in the nearest `conftest.py`:
- **`tests/conftest.py`** (global) — the `frame_factory` solid-RGB frame builder and the
  in-memory DB fixtures (`engine`/`session_factory`/`image_dir` + `detection_factory`, which
  inserts a row and its on-disk image/thumbnail/video stubs). These are global rather than
  under `tests/db/` because both the `db` and `api` suites need them and sibling `conftest.py`
  files aren't visible across packages. (`onnxruntime` is a real dependency — installed in the
  project `.venv` and pinned on the Pi/CI — so `ml` imports resolve normally; the classifier
  tests still skip when the ONNX *model file* is absent.)
- **`tests/api/conftest.py`** — `make_client`/`client` (`TestClient` with the DB + image
  deps overridden), `seeded_detections`, and the `FakeHttpxResponse` stand-in for the
  detector-proxy routes.
- **`tests/ml/conftest.py`** — injectable pipeline fakes (`FakeDetection`, `RecordingWriter`,
  `RecordingRecorder`, and a `stable_tracker` builder) for the classification-pipeline tests,
  plus the two **real-model** fixtures used by the end-to-end tests: `bird_image_cases`
  (loads the labelled `tests/_test_images/` JPEGs + their boxes into `ImageCase`
  tuples — `skip`s if the manifest/images are absent) and `real_classifier` (builds the
  int8 ONNX `Classifier` via `setup_classifier` — `skip`s if the model file is absent).
  Both are module-scoped so the large JPEGs / ONNX session are loaded once per module.
Camera/crop fakes stay local to their single test file (`test_camera_server.py`,
`test_crop_controller.py`) since they are not shared and are constructed at parametrize
time (before fixtures exist).

**End-to-end detection tests** — `tests/_test_images/` holds a small labelled fixture
set (`Erithacus_rubecula.jpg`, `Eurasian_jay.jpg`) plus `bounding_box_locations.json`
(each entry: `image` repo-relative path, `species` = the **classifier label** the crop
should predict, and `bounding_box` = `[[tlx, tly], [brx, bry]]`). The JSON is tracked via
a `!tests/_test_images/bounding_box_locations.json` negation of the repo-wide `*.json`
ignore; the JPEGs are committed so the tests are runnable off the Pi (the `.onnx` model
stays out-of-band, so the tests `skip` where it is absent). The hand-labelled boxes stand
in for **Stage 1** (the IMX500 `.rpk` object detector, which only runs on the camera
silicon and cannot execute off-Pi); everything downstream runs for real:
- **`tests/ml/test_bird_species_classification.py`** — classifier accuracy: crops each
  box with the live `preprocess_roi`, classifies with the real int8 model, and asserts the
  prediction equals the manifest's `species`.
- **`tests/ml/test_end_to_end_detection.py`** — the full seam from classifier to DB: a
  `Detection` built from the JSON box flows through `ClassificationManager` (stable-track
  gating, sync dispatch) → real classifier → `_persist_detection`, writing the still +
  thumbnail to a tmp `IMAGE_DIR` (via `monkeypatch` of `classification_pipeline.IMAGE_DIR`)
  and a `DetectionRecord` through the real `DetectionWriter` into the in-memory SQLite
  `engine`/`session_factory`. Asserts the row (species, `detection_confidence`, normalized
  box, `no_video_reason`) and the on-disk files. A second test feeds a zero-area box and
  asserts the empty-ROI guard skips it (no row; the track left unclassified).

### Type checking (mypy)

`mypy.ini` at the repo root configures mypy. Because everything is one package with
absolute imports, mypy runs in a **single pass**:

```bash
mypy birdscanner tools tests --check-untyped-defs
```

`mypy.ini` silences missing-import noise for the Pi-only native bindings (`libcamera`,
`picamera2`) and the untyped third-party libs (`onnxruntime`, `psutil`, `paramiko`, `PIL`
— Pillow ships no `py.typed` marker, plus `onnx` and `matplotlib` which are used only by
the dev-only scripts under `tools/`), so mypy reports only genuine type errors in our own
code.

### Linting (pylint, black)

Black and pylint need to be run for linting and formatting. 

Always run black first:

```bash
black birdscanner tools tests
```

Pylint rates the codebase **10.00/10** (exit 0). It runs in two invocations only because
the tests use a separate rcfile to relax pytest idioms — not for import resolution, which
is now uniform:

```bash
pylint birdscanner tools                    # source tree (root .pylintrc)
pylint tests --rcfile=tests/.pylintrc        # tests (test-tuned config)
```

- **`.pylintrc`** (root) lints the source tree. It silences the same Pi-only / untyped
  native libs mypy ignores via `ignored-modules` (`libcamera`, `picamera2*`,
  `onnxruntime*`, `onnx`) and sets `generated-members=cv2.*` (OpenCV's members are
  populated dynamically, so pylint can't see them). Only a small set of opinionated
  checks are disabled project-wide (`too-few-public-methods`, `import-outside-toplevel`
  — the deliberate lazy Pi-only imports, `broad-exception-caught`, `global-statement`
  — the module-singleton pattern); `max-line-length=100`, `max-args=5`,
  `max-attributes=15`. **The complexity/argument checks are deliberately left enabled**
  (`too-many-arguments`/`-positional-arguments`/`-locals`/`-branches`/`-statements`/
  `-instance-attributes`, `duplicate-code`): rather than disabling them, wide signatures
  are collapsed into small **parameter objects / value objects** and god-functions are
  **decomposed** (see below). Do not add a `# pylint: disable=too-many-*` to silence one
  of these — introduce a cohesive object or split the function instead. The only two
  in-code suppressions left are genuine false positives, written as **block** disables
  (so `black` re-wrapping can't drift them off their anchor line): the side-effect
  `DetectionRecord` import in `db/database.py` and the `wrong-import-position` in
  `tests/tools/test_build_species_reference.py`.
  - Parameter/value objects introduced for this: `ml/detection_utils.Box` (all box
    geometry — `iou`/`preprocess_roi`/`normalized_box` are thin tuple wrappers over it),
    `ml/classification_pipeline.PipelineContext` (the per-detection deps, shared by the
    processing functions **and** `ClassificationManager.__init__`) plus `Still` /
    `Classification`, `db/writer.DetectionWriter.write(record)` (takes a pre-built
    `DetectionRecord` instead of 14 kwargs), `detector/config`'s nested
    `IntrinsicsConfig` / `VideoConfig`, `detector/crop.NormalizedBox` /
    `SensorDimensions`, `detector/crop_controller.CropControllerConfig`,
    `api/routers/detections.DetectionFilters` (a FastAPI `Depends` class), and
    `tools/build_species_reference.BuildOptions`.
  - Decomposed god-functions: `detector/main.main` (now a short startup script:
    `configure_logging` → `init_db` → `wait_for_camera`/`prepare_intrinsics`/`build_camera`
    (from `detector/camera.py`) → `build_gating`/`build_manager` (from `detector/gating.py`)
    → `_start_control_server` → `_run_capture_loop`, threading state through the `Camera` /
    `Gating` bundles). The camera bring-up and classification-pipeline wiring now live in
    their own cohesive modules (`detector/camera.py`, `detector/gating.py`) so `main.py` is
    just the entry point plus the per-frame loop. `detector/camera_server` (the request
    handler is a module-level `CameraRequestHandler` reading its deps off a `_ControlServer`,
    not a per-server closure).
- **`tests/.pylintrc`** additionally relaxes pytest idioms (`missing-*-docstring`,
  `redefined-outer-name` from fixtures, `unused-argument`, `protected-access`,
  `use-implicit-booleaness-not-comparison` for exact-empty-literal assertions). It exists
  only to relax these idioms — imports resolve uniformly now. Run it explicitly with
  `--rcfile=tests/.pylintrc`.

## Architecture

### Detection pipeline (frame loop)

```
IMX500 on-chip inference
  └─ parse_detections()         # extract Detection objects from inference tensor
       └─ StableDetectionTracker.update_frame()   # IoU-based multi-frame tracking
            └─ process_detections() [picam2 pre_callback]
                 └─ ClassificationManager.process()
                      └─ process_single_detection_with_stable_tracks()  # new path
                           └─ BestFrameSelector.take(track_id)  # best frame seen across the track
                           └─ Classifier.classify()  # ConvNeXt V2 Tiny ONNX
                                └─ save *clean* full image + 200×200 thumbnail to IMAGE_DIR/{species}/
                                └─ VideoRecorder.trigger()  → short mp4 clip (pre+post-roll) to IMAGE_DIR/{species}/
                                └─ DetectionWriter.write()  → SQLite detections table (incl. normalized box + video_path)
```

The still image and thumbnail come from the **best frame observed across the track**
(highest YOLO detection confidence), not the arbitrary frame that triggered
classification — see `birdscanner/ml/best_frame.py`. Alongside the still, a short
**mp4 clip** is saved per detection (`birdscanner/detector/video_recorder.py`); the
clip finishes encoding a few seconds *after* the DB row is written, so its file (and
the `/video` endpoint) briefly 404s while the UI falls back to the still.

The saved full image is the **raw frame — no bounding box is drawn on it** (the box
is only drawn on the live preview stream). The detection box is persisted to the DB as
normalized `[0, 1]` coordinates (`box_x/y/w/h`) so the frontend can overlay it on the
clean image with an on/off toggle (see `frontend/src/components/Lightbox.tsx`).

### Package layout

The whole codebase is one package, `birdscanner/`, run via `python -m ...` with absolute
imports. It has four layered subpackages plus the consolidated assets and dev tooling:

```
birdscanner/
  detector/   Pi-only camera + hardware control and the entry point (main, camera,
              gating, config, camera_server, crop, crop_controller, track_logging,
              paths, video_recorder)
  ml/         platform-independent inference (object_detection, detection_utils,
              tracking, classification, classification_pipeline, best_frame)
  api/        FastAPI REST API + routers (was backend/)
  db/         SQLite persistence
assets/       labels/ (coco_labels.txt), models/ (classifier ONNX + class map),
              species_reference/ (data bank)
tools/        dev / offline tooling — NOT runtime (build_species_reference, quantize,
              camera_smoke*)
```

Import direction is one-way: `detector → ml → db` and `api → db`. `ml/` must never import
from `detector/` or `api/`.

### Key modules

**`birdscanner/ml/object_detection.py`** — core object detection only:
- `Detection` — bounding box + category + confidence; box is set after coordinate conversion via `imx500.convert_inference_coords`
- `parse_detections` — extracts `Detection` objects from the IMX500 inference tensor; `get_labels` filters the intrinsics label list
- `last_detections` — module-global fallback returned when a frame yields no inference output

**`birdscanner/ml/detection_utils.py`** — stateless geometry/drawing helpers shared across the pipeline:
- `Box` — the box **value object** that owns all geometry (`.x2`/`.y2`/`.area`, `.iou(other)`, `.padded_square(img_w, img_h)`, `.crop(image)`, `.normalized(shape)`). Centralising the maths here is what keeps the callers under the `too-many-locals` limit. The module functions below are thin, tuple-based wrappers over it so existing call sites keep passing plain `(x, y, w, h)` tuples
- `iou` — Intersection-over-Union for two `(x, y, w, h)` boxes (delegates to `Box.iou`)
- `preprocess_roi` — expands a box to a padded square and crops the ROI (delegates to `Box.padded_square`/`.crop`)
- `draw_boxes(image, coords, detection, labels, classification=None)` — annotates a frame with a box + label + optional `(species, confidence)` classification result (grouped into one param to stay under `too-many-arguments`). Used only for the **live preview** stream now — it is no longer drawn onto the saved image (the box is persisted instead, see `normalized_box`)
- `normalized_box` — converts a `(x, y, w, h)` pixel box to fractions of the image dimensions, clamped to `[0, 1]` (delegates to `Box.normalized`); persisted with each detection so the UI can overlay the box on the clean saved image at any rendered size
- `save_thumbnail` — writes a 200×200 JPEG thumbnail
- `label_for_category` — bounds-checked label look-up; returns `None` when the class index is outside the label list. The IMX500 SSD model occasionally emits a spurious detection whose category index is out of range, so every `labels[category]` access goes through this helper. `draw_boxes` falls back to an `id:<n>` placeholder, and `process_detections` skips the detection entirely (logging a warning to the `tracking` logger) instead of crashing the camera callback with an `IndexError`

**`birdscanner/ml/tracking.py`** — multi-frame stability tracking:
- `StableDetectionTracker` — IoU-based tracker; a detection must match across `min_stable_frames` consecutive frames before `should_run_bird_classification_for_detection` returns `True`; each track is classified at most once (`mark_classified`)
- `StableTrack`, `match_detection_to_track`, `update_tracks_for_frame`, `should_classify_track` — the underlying pure-function tracking primitives (directly unit-tested in `tests/ml/test_tracking.py`)
- `stable_detection_tracker` — module-global default tracker instance

**`birdscanner/ml/classification_pipeline.py`** — classification orchestration + persistence:
- `PipelineContext` — the **parameter object** bundling every per-detection dependency (`classifier`, `tracker`, `classify_fn`, `detection_writer`, `best_frame_selector`, `record_fn`, `video_frame_fn`); its `__post_init__` fills `tracker`/`classify_fn` from the module defaults. Built once in `main._build_manager` and shared by `ClassificationManager` **and** the processing functions, so neither has a long argument list. `Still` (frame + box) and `Classification` (species + confidence) are small `NamedTuple` value objects that keep the persistence helpers under the argument/local limits
- `ClassificationManager(context, *, use_multithreading=False, queue_maxsize=0, use_stable_track_gating=False)` — wraps sync/async (threaded `Queue`) dispatch; in async mode items are dropped if the queue is full so the camera callback never blocks; all pipeline deps come from the injected `PipelineContext` (so `ml/` never imports picamera2). Both the sync and async paths run through `_dispatch`, which **catches and logs any exception from processing a single detection** so one bad detection can never take down the pipeline: an unhandled exception in the async worker thread used to kill it permanently — after which every queued detection silently piled up unprocessed and **nothing was ever classified or saved again** (the symptom was tracks becoming stable but staying `species=None`, with no new DB rows / gallery entries). The async `_worker_loop` also `task_done()`s in a `finally` so the queue never wedges
- `process_detections` — picam2 `pre_callback` entry point; draws boxes, feeds every clean frame to `context.video_frame_fn` (the pre-roll buffer), offers each bird detection to the `context.best_frame_selector` (keyed by its track), and queues bird detections
- `process_single_detection_with_stable_tracks(item, context, results_lock)` — new gating path; `process_single_detection` is the legacy per-frame IoU-cache path (kept for reference). Decomposed into small helpers (`_best_still`, `_classify_track`, `_save_still_and_thumbnail`, `_start_clip`, `_persist_detection`) to stay under the complexity limits. At the stable trigger it takes the track's **best frame** from `best_frame_selector` (falling back to the trigger frame) and uses it for the classification ROI, the saved **raw** still (no box drawn), the thumbnail, and the persisted normalized box; `_persist_detection` builds a `DetectionRecord` and calls `DetectionWriter.write(record)`. It also calls `record_fn` to start a clip: `_start_clip` returns a `(video_path, no_video_reason)` pair, so `video_path` is persisted only when recording actually began and, when it didn't, `no_video_reason` records **why** (`"recorder_busy"` for a single-flight-declined trigger, `"disabled"` when no recorder is wired) — the UI reads this to explain the greyed-out Video toggle. **Skips classification for a degenerate (zero-area) detection box** whose ROI is empty (`roi.size == 0`) — feeding an empty array to the classifier raises inside PIL (`Image.fromarray`); the track is left unclassified so a later, non-degenerate frame can still classify it. This was the most common trigger of the worker-death bug above
- `setup_classifier`, `run_bird_classification`, `update_detection_classifications_cache`, `classification_results`
- `IMAGE_DIR` — root directory for saved images, sourced from the `IMAGE_DIR` env var (defaults to `/home/stefan/Pictures/bird_detections`)

**`birdscanner/ml/best_frame.py`** — per-track best-frame selection (pure numpy, unit-tested in `tests/test_best_frame.py`):
- `BestFrameSelector` — thread-safe (`observe`/`take`/`discard`) store that keeps the single highest-scoring `(frame, box)` per `track_id` (≤1 frame retained per active track). Written on the camera thread (`observe`), read on the classifier thread (`take`), and freed when a track ends via the tracker's `on_track_deleted` callback (`discard`). The score is the detection's **YOLO confidence** (`detection.conf`) — zero extra compute. `BestCandidate` is the retained `frame`/`box`/`score` dataclass

**`birdscanner/api/`** — FastAPI REST API (Phase 2):
- `birdscanner/api/main.py` — app factory; mounts the API routers; optionally serves `frontend/dist/` at `/` when the build exists via `SPAStaticFiles`, a `StaticFiles` subclass whose `get_response` falls back to `index.html` on a 404. The React app uses client-side routing (`BrowserRouter`), so deep links like `/history` have no file on disk; a plain `StaticFiles` mount returned `{"detail":"Not Found"}` on direct navigation/refresh. The fallback serves `index.html` so the SPA loads and renders the route client-side. API routes are unaffected (registered before the mount, so they take precedence)
- `birdscanner/api/dependencies.py` — `get_session()`, `get_image_dir()`, and `get_reference_dir()` FastAPI dependency providers (reads `DB_PATH` / `IMAGE_DIR` / `SPECIES_REFERENCE_DIR` env vars); the engine is opened **read-only** (`make_engine(read_only=True)`) and the API never runs `init_db` — the detector owns schema creation, and the DB is mounted read-only. `get_reference_dir()` defaults to the repo-relative `assets/species_reference/` and may point at a non-existent dir — the reference API degrades gracefully when the manifest is absent
- `birdscanner/api/routers/detections.py` — `GET /api/detections` (paginated + filtered by `species`, `from`, `to`, and `min_confidence` — a 0–1 floor applied as `confidence >= min_confidence` against the **classification** `confidence` (not `detection_confidence`)). The filter query params are grouped into a `DetectionFilters` FastAPI `Depends` class (with an `.apply(query)` method) so the endpoint signature stays short; `limit`/`offset`/`session` remain direct params. `GET /api/detections/{id}`, and `DELETE /api/detections/{id}`. List results are ordered by `timestamp` **desc, then `id` desc** — the `id` tiebreaker keeps tied-timestamp rows in a deterministic order across paginated requests so offset-based pages don't overlap (overlap surfaced duplicate detections in the History tab). Since the API mounts the DB + images read-only, the delete is **proxied** to the detector's control server (`${DETECTOR_URL}/detections/{id}`, same channel as the camera snapshot) which owns the read-write data volume; relays 204 on success, 404 if the row is missing, 503 if the detector is unreachable
- `birdscanner/api/routers/images.py` — `GET /api/images/{id}/thumbnail`, `GET /api/images/{id}/full`, `GET /api/images/{id}/video` (mp4 clip; 404 when the row has no `video_path` or the file isn't on disk yet — the clip finishes encoding a few seconds after the row is written), `GET /api/images/download?ids=...` (chunked ZIP of the stills)
- `birdscanner/api/routers/system.py` — `GET /api/system` (CPU/mem/disk/temp/uptime via psutil)
- `birdscanner/api/routers/network.py` — network monitoring + on-demand speed test (prefix `/api/network`). **Passive usage graph**: a module-level `_Sampler` daemon thread (started at import, mirroring `system._BOOT_TIME`) reads the host NIC byte counters (`psutil.net_io_counters`) every `_SAMPLE_INTERVAL_SEC=3` s into a ~1 h ring buffer, deriving per-sample download/upload **kbps** from the counter delta (the first reading only seeds the baseline). `GET /api/network/history?range=5m|30m|1h` returns `{interval_sec, samples:[{t, rx_kbps, tx_kbps}]}` for the window (400 on an unknown range). Reading counters transfers nothing, so the sampler runs continuously at no bandwidth cost. **Active speed test**: `POST /api/network/speedtest` calls `run_speed_test()` which downloads `SPEEDTEST_DOWNLOAD_BYTES` (~1 MB) from and uploads `SPEEDTEST_UPLOAD_BYTES` (~256 KB) to Cloudflare's `__down`/`__up` endpoints (overridable via `SPEEDTEST_DOWN_URL`/`SPEEDTEST_UP_URL`) and reports measured Mbps each way + bytes transferred; 503 when the endpoint is unreachable. Payloads are deliberately small to spare the Pi's limited connection — the test is a manual button, **never polled**. `run_speed_test` is a module function so tests monkeypatch it without hitting the network
- `birdscanner/api/routers/species.py` — `GET /api/species` (list with counts, sorted by count desc)
- `birdscanner/api/routers/reference.py` — species **reference / comparison** data (prefix `/api/species`, tags `["reference"]`): `GET /api/species/{name}/reference` returns the cached Wikipedia/Wikidata info for a species (`common_name`, `scientific_name`, `summary`, `behaviour`, `wikipedia_url`, and an `images` list of `{url, attribution, license}`) — **404** when the species isn't in the manifest; `GET /api/species/{name}/reference/images/{index}` serves the locally-cached reference JPEG (path-traversal-guarded, 404 on out-of-range/missing). Reads the manifest from `get_reference_dir()` via a per-dir memoized loader; a missing/invalid `manifest.json` is treated as "no references" so every lookup 404s cleanly instead of crashing. The served image `url` points back at the images endpoint — the on-disk `path`/upstream `source_url` are never exposed. Powers the gallery/dashboard comparison panel
- `birdscanner/api/routers/camera.py` — proxies the detector's snapshot + crop-control endpoints (`${DETECTOR_URL}`, default `http://detector:8000`) via httpx, since the API can't open the camera itself (the detector owns it exclusively + the data volume is read-only): `GET /api/camera/snapshot` → `/capture` (cropped feed), `GET /api/camera/snapshot/full` → `/capture/full` (full-sensor preview for the crop editor), `GET /api/camera/crop` → `/crop`, `POST /api/camera/crop` → `/crop` (relays the detector's status, e.g. 400 for a bad box). Returns 503 when the detector is unreachable

**`birdscanner/db/`** — SQLite persistence layer (Phase 1):
- `birdscanner/db/models.py` — `DetectionRecord` SQLModel ORM model (`detections` table). Stores **two** confidences: `confidence` is the species-classification score (ConvNeXt), and the nullable `detection_confidence` is the object-detection score (YOLO11n on the IMX500) for the bird box (null for legacy rows written before it was persisted). Also includes nullable `box_x/box_y/box_w/box_h` floats — the detection box in normalized `[0, 1]` image fractions (null for legacy rows written before boxes were persisted) — and a nullable `video_path` (the saved mp4 clip, relative to `IMAGE_DIR`; null for legacy rows, when video is disabled, or when a single-flight-declined trigger produced no clip). When `video_path` is null the nullable `no_video_reason` string says **why** the clip is missing — `"recorder_busy"` (the single-flight recorder was busy with another sighting) or `"disabled"` (video recording is off); null when a clip exists or for legacy rows. The UI surfaces this as the tooltip on the disabled Video toggle
- `birdscanner/db/database.py` — `make_engine()` / `init_db()` / `make_session_factory()`; DB path from `DB_PATH` env var. `make_engine(read_only=True)` opens the SQLite file via the `mode=ro` URI so it can be read off a read-only mount without attempting to create a journal file (used by the API). `init_db()` also runs `_migrate_detections_columns()` — a lightweight `ALTER TABLE ... ADD COLUMN` pass that backfills nullable columns added after the initial schema (e.g. `detection_confidence`, `video_path`, `no_video_reason`, `box_*`), since SQLModel's `create_all` never alters an existing table. Driven by the `_DETECTIONS_ADDED_COLUMNS` map; add new post-launch columns there
- `birdscanner/db/writer.py` — `DetectionWriter`: fire-and-forget background-thread writer; `write(record)` takes a **pre-built `DetectionRecord`** (the caller names each column via the model's keyword-only constructor) and enqueues it; `stop()` flushes and exits. The writer no longer mirrors the column list in its own signature (that was `too-many-arguments`/`-locals`)
- `birdscanner/db/deleter.py` — `delete_detection(session_factory, image_dir, id)`: synchronously deletes a detection's DB row **and** its image + thumbnail + video files (best-effort file unlink — a missing file does not block the row delete; the nullable `video_path` is skipped when absent); returns `False` when no row with that id exists. Runs in the detector (which owns the read-write data volume); the API proxies delete requests to it. Unlike the writer it is synchronous (it runs on the control server's HTTP thread, not the camera callback)
- `birdscanner/db/migrations/001_initial.sql` — plain SQL migration (reference; `init_db()` is authoritative)

**`tools/build_species_reference.py`** — offline species-reference data-bank builder (**not** part of the runtime; run on a dev machine with internet):
- Iterates every label in `assets/models/convnext_v2_tiny.onnx_class_to_idx.json` (the source of truth for "what species exist"; the `"Unknown"` sentinel is skipped) and fetches, per species: the **Wikipedia** REST summary (`summary` + lead image + canonical URL + Wikidata id), a best-effort "Behaviour"/"Habitat" section (`behaviour`) via the MediaWiki API, image attribution/license via `imageinfo`+`extmetadata`, and the **Wikidata** taxon name (P225) for `scientific_name`. Pure stdlib (`urllib`), descriptive `User-Agent`, polite throttling
- Writes `assets/species_reference/manifest.json` (schema: `{version, generated_at, source, species:{<label>:{common_name, scientific_name|null, summary, behaviour|null, wikipedia_url|null, images:[{path, source_url, attribution, license|null}]}}}`) + `coverage_report.json` (buckets: `missing`, `no_images`, `no_scientific_name`, `skipped`), and downloads each reference image to `assets/species_reference/images/<slug>/<n>.jpg` (`<slug>` = label lowercased, non-alphanumeric runs → single `-`)
- **Incremental / idempotent**: re-runs only fetch labels missing from the manifest (or whose image files are gone), so adding a new class = add the label to the classifier's `class_to_idx.json` + re-run. Flags: `--force` (refetch all), `--limit N` (only the first N labels needing a fetch — for a quick live smoke test), `--throttle SECONDS`; these are bundled into a `BuildOptions` dataclass passed to `build_manifest(labels, overrides, existing, options)`
- `assets/species_reference/overrides.json` — hand-maintained `{<label>:{wikipedia_title?, scientific_name?, skip?}}`; consulted first, so it fixes name→article mismatches (e.g. apostrophe-stripped labels like `Audouins gull` → `Audouin's gull`) and forces `skip` for `"Unknown"`. This file is **source** (curate + commit it); the generated `manifest.json`/`coverage_report.json` and the downloaded `images/` are **build artifacts** shipped/mounted out-of-band (like the ONNX model — a full build is ~hundreds of MB of JPEGs), and are `.gitignore`d (a `!assets/species_reference/overrides.json` negation keeps the curated overrides tracked despite the repo-wide `*.json` rule). The API finds the data via `SPECIES_REFERENCE_DIR` (default `assets/species_reference/`). **In the Docker stack** the bank is **not** baked into the api image (`Dockerfile.api` copies only `birdscanner/` + `frontend/dist`); instead it lives on the data volume at `/data/species_reference` with `SPECIES_REFERENCE_DIR=/data/species_reference` (set in `.env.example`). Build it on a dev machine and copy `manifest.json` + `images/` into `<data-volume>/species_reference/` on the host (e.g. `/mnt/ssd/birdfinder-data/species_reference/`); the api mounts the data volume read-only and reads it there. If this dir is missing/empty, **every** reference lookup 404s and the Lightbox panel shows "No reference available" for all species — that is the symptom of an un-deployed bank, not a code bug

**`birdscanner/ml/classification.py`** — ONNX classifier stack:
- `ONNXClassifier` — raw ONNX Runtime wrapper; expects `(N, C, H, W)` float32
- `Classifier` — adds preprocessing and class-index mapping; `classify()` returns `(species_str, confidence_float)`
- `build_preprocessing` — pure PIL+NumPy pipeline (resize → center crop → ImageNet normalize → NCHW)

**`birdscanner/detector/config.py`** — application configuration:
- `Config` dataclass + module-level `config` instance; edit values here instead of passing flags. To keep the top-level dataclass small (`too-many-instance-attributes`), the settings are grouped into two nested sub-configs: **`config.intrinsics`** (`IntrinsicsConfig`: the network-intrinsics overrides `main.py` pushes onto the IMX500 intrinsics object — `fps`, `bbox_normalization`, `bbox_order`, `ignore_dash_labels`, `preserve_aspect_ratio`, `labels`) and **`config.video`** (`VideoConfig`: `save` default `True`, `pre_roll_seconds` default `3.0`, `post_roll_seconds` default `4.0`). The remaining top-level fields are `model`, `threshold`, `print_intrinsics`, `multithread`, `object_duration_threshold`, `debug`, `preview`. `main.py` imports the instance as `app_config` (to avoid clashing with the local `config` returned by `picam2.create_preview_configuration`) and its intrinsics-override loop iterates `vars(app_config.intrinsics)`. The video/best-frame machinery is only wired when stable-track gating is active (`object_duration_threshold > 0`)

**`birdscanner/detector/paths.py`** — package-anchored data-file resolution:
- Resolves the detector's on-disk data files relative to the **package location** (`<repo>/assets`, `/app/assets` in Docker), not the current working directory — this is what lets the detector run as `python -m birdscanner.detector.main` from anywhere (the old `cwd=src` requirement is gone). `assets_dir()` / `model_dir()` are overridable via the `ASSETS_DIR` / `MODEL_DIR` env vars (default `assets/` and `assets/models/`). Helpers `coco_labels_path()`, `class_to_idx_path()`, `classifier_model_path()` return the concrete files; `main.py` uses them instead of hardcoded relative strings

**`birdscanner/detector/main.py`** — entry point + capture loop only:
- `main()` reads as a short startup script (`configure_logging` → `make_engine`/`init_db` → `wait_for_camera`/`prepare_intrinsics` → `setup_classifier` → `build_gating` → `DetectionWriter`/`build_manager` → `build_camera` → `_start_control_server` → `_run_capture_loop`). The camera bring-up lives in `detector/camera.py` and the classification-pipeline wiring in `detector/gating.py`; `main.py` keeps only `_start_control_server`, `_run_capture_loop`, `_shutdown`, and `main`. A `KeyboardInterrupt` out of `_run_capture_loop` triggers `_shutdown`
- Reads all runtime settings from `config.config` (imported as `app_config`); no `argparse`
- `_run_capture_loop` installs the per-frame `detection_callback` (updates the tracker + queues bird detections via `process_detections`) then loops: it holds the `CropController`'s `camera_lock` across `capture_metadata()` + `parse_detections()` so a UI-triggered reconfigure (which may `stop()`/`start()` the camera) never races an in-flight capture, and calls `update_detection_classifications_cache` each frame to keep the legacy temporal filter in sync alongside the new tracker
- Creates the SQLite engine and runs `init_db()` on startup **before camera init**, so the schema always exists; the detector owns all DB writes, so this lets the read-only API serve an empty gallery even when the camera never comes up. The same `engine` is reused to wire a `DetectionWriter` into the `ClassificationManager` (via `build_manager`) so every high-confidence classification is persisted, and to bind the delete callback for `_start_control_server`; the writer is flushed via `detection_writer.stop()` on `KeyboardInterrupt`
- `_start_control_server` wires the `camera_server` (snapshots + crop + detection deletes) to the started camera and the DB engine; a port-bind `OSError` is logged and swallowed so the detection pipeline keeps running without it

**`birdscanner/detector/camera.py`** — camera bring-up (IMX500 + Picamera2 + crop):
- `wait_for_camera()` wraps `IMX500(...)` in a retry-with-backoff loop (30 s default): when the camera dev-node is missing it logs a concise warning and retries instead of crashing, so the detector never crash-loops and auto-recovers when the camera reappears
- `prepare_intrinsics(imx500)` validates the object-detection task and iterates `vars(app_config.intrinsics)`, so optional intrinsic fields left as `None` do not clobber the network intrinsics defaults (and `print_intrinsics` prints + exits)
- `build_camera(imx500, intrinsics)` loads the persisted crop region via `load_crop_region(crop_config_path(), default_crop_region(...))` (default = the historical 900×900 at `(4/13, 5/10)` of the 4056×3040 sensor, aimed at the feeder), sizes the `main` stream to the crop's aspect ratio (`main_stream_size_for_crop`), applies `vflip=True, hflip=True` (camera is mounted upside-down), starts the camera, and builds the `CropController`. The camera config is built through a `build_camera_config(main_size, scaler_crop)` closure that centralises every picamera2 knob so `CropController` can rebuild it on a reconfigure. Returns a `Camera` bundle (picam2 + imx500 + intrinsics + crop_controller). This module is Pi-only (imports `libcamera`/`picamera2`); no test imports it

**`birdscanner/detector/gating.py`** — classification-pipeline wiring:
- `build_gating(intrinsics)` builds the stable-track gating machinery and returns a `Gating` bundle (`use_stable_tracks`, tracker, best-frame selector, video recorder). When `object_duration_threshold <= 0` gating is disabled (legacy per-frame mode) and every component is `None`. Otherwise it constructs the `StableDetectionTracker` (wired to a `TrackingLogger` for lifecycle logging), a `BestFrameSelector` freed per-track via the tracker's combined `on_track_deleted` callback (a non-optional local captured by that closure so it never dereferences a possibly-`None` selector), and — when `save_video` is set — a `VideoRecorder`. `min_stable_frames(fps)` converts `object_duration_threshold` × fps into a frame count
- `build_manager(classifier, gating, detection_writer)` assembles the `PipelineContext` and `ClassificationManager` from the `Gating` bundle; the recorder's `add_frame`/`trigger` are injected as `video_frame_fn`/`record_fn` (keeping picamera2 out of `ml/`). It installs the manager's results lock. Imports only `ml/`, `config`, `track_logging`, `video_recorder`, and `db/writer` — no picamera2, so it is importable without a camera

**`birdscanner/detector/video_recorder.py`** — on-demand short-clip recorder (unit-tested in `tests/test_video_recorder.py`):
- `VideoRecorder` keeps a bounded `deque` of recent raw `main`-stream frames (the pre-roll) fed by `add_frame` every frame — **cheap, no encoding while idle**, which matters because the **Pi 5 has no hardware video encoder** so all encoding is software (CPU). `trigger(dest_path)` snapshots the pre-roll, keeps collecting `post_roll_seconds` of live frames, then encodes the whole sequence to an mp4 (`cv2.VideoWriter`, `mp4v`, RGB→BGR like the still writes) on a **background thread** so the camera callback never blocks. **Single-flight**: a trigger while a clip is recording is declined (returns `False`), bounding CPU/RAM. Durations come from `config` (`video_pre_roll_seconds`/`video_post_roll_seconds`); fps from the camera inference rate. Trade-off: the RAM buffer costs ~1.2 MB/frame × pre-roll frames — downscale/limit if it bites

**`frontend/`** — React + Vite + Tailwind dashboard (Phase 3 & 4):
- **Design system ("field journal" theme)** — the UI is themed as a naturalist's printed bird guide rather than a generic dashboard. Tokens live in `frontend/tailwind.config.ts`: a warm-paper palette (`paper` page bg, `card` lifted stock, `line` tea-stain hairlines, `ink` forest-green text, `bark` secondary text, `sage`/`sage-deep` structural green, `gold`/`gold-deep` the single goldfinch-ochre accent, `rust` for destructive actions only) plus two fonts (`font-display` = Fraunces, used for headings + species names; `font-sans` = Hanken Grotesk for body/UI) and `shadow-plate`/`shadow-plate-lift`. `frontend/index.html` loads both fonts from Google Fonts. `frontend/src/index.css` sets the paper base (a near-invisible dotted grain), the on-brand `:focus-visible` ring, a `prefers-reduced-motion` guard, warm scrollbars, and two reusable utilities: `.eyebrow` (the ruled small-caps section label used as section/page eyebrows) and `.tnum` (tabular numerals — replaces the old `font-mono` for all readings). The single accent is `gold`; numbers use `.tnum`, never mono. When adding UI, reuse these tokens/utilities instead of raw Tailwind `slate-*`/`emerald-*` colours
- `frontend/src/api.ts` — typed fetch wrappers for all `/api/*` endpoints (including `detections.delete(id)` → `DELETE /api/detections/{id}`); exports `Detection` (incl. both `confidence` (classification) and the nullable `detection_confidence` (YOLO) scores, the nullable normalized `box_x/box_y/box_w/box_h` fields used to overlay the detection box in the Lightbox, the nullable `video_path` flag indicating a clip exists, and the nullable `no_video_reason` explaining why a clip is absent), `SystemStatus`, `SpeciesSummary`, `CropState`, `NormalizedBox`, `SpeciesReference`, `SpeciesReferenceImage`, `NetworkSample`, `NetworkHistory`, `NetworkRange`, `SpeedTestResult` interfaces plus `timeAgo` and `formatUptime` helpers. `api.images` exposes `thumbnailUrl`, `fullUrl`, `videoUrl` (`GET /api/images/{id}/video`), and `downloadUrl`. `api.camera` exposes `snapshotUrl`, `fullSnapshotUrl`, `getCrop`, and `setCrop` (POSTs a normalized box or `{reset:true}`); `api.species.reference(name)` fetches a species' reference data; `api.network` exposes `history(range)` (`GET /api/network/history`) and `speedTest()` (`POST /api/network/speedtest`). `apiFetch` throws an `ApiError` carrying the HTTP `status` so callers can distinguish a 404 (e.g. "no reference yet") from a real error
- `frontend/src/App.tsx` — root component; sets up `react-router-dom` `BrowserRouter` with routes `/` → Dashboard, `/history` → History, `/camera` → Camera, and `/hardware` → Hardware; renders a top-level nav bar
- `frontend/src/components/SystemMonitor.tsx` — polls `/api/system` every 5 s; renders animated gauge bars (green/yellow/red) for CPU, memory, disk, temp, and uptime. The full gauge layout is **always rendered** (placeholder `—` values + empty bars until the first poll resolves) so the card keeps a fixed size from the initial render — previously it showed a single "Loading…" line and the whole page shifted down when values arrived. The CPU Temp gauge is therefore always present (shows `—` when temp is unavailable) rather than conditionally mounted. Lives on the **Hardware** page (`/hardware`), no longer on the Dashboard
- `frontend/src/components/NetworkMonitor.tsx` — the Hardware page's network panel. **Usage graph**: polls `api.network.history(range)` every 3 s and draws a hand-rolled SVG line+area chart of download (`rx`, sage) and upload (`tx`, gold) throughput, with a 5m/30m/1h window toggle. The x-axis is anchored to the live `[now − window, now]` span so the trace grows in from the right; the y-axis auto-scales to a "nice" ceiling via `niceCeil`; rates format adaptively as Kbps/Mbps (`formatRate`). The SVG uses theme `fill-*`/`stroke-*` utilities (`stroke-sage-deep`, `stroke-gold-deep`, `fill-sage/15`, `fill-gold/15`, `stroke-line`). A `rangeRef` keeps the poll interval from tearing down on every window change; a second effect refetches immediately when the range toggles so it feels responsive. **Speed test**: a manual "Test network" button calls `api.network.speedTest()` and shows the measured down/up Mbps + bytes transferred + run time; never auto-polled (it costs bandwidth)
- `frontend/src/pages/Hardware.tsx` — Hardware page (`/hardware`); composes `SystemMonitor` (The Station, moved here off the Dashboard) above `NetworkMonitor` (usage graph + speed test)
- `frontend/src/components/DetectionCard.tsx` — loads thumbnail from `/api/images/{id}/thumbnail`; shows species, both confidences (`{confidence}% match` for the species classification plus `{detection_confidence}% spotted` for the YOLO object detection, the latter omitted for legacy rows where it is null), and a time-ago label; supports optional `onSelect`/`selected` props for bulk-select mode and `onOpenLightbox` prop to trigger the lightbox
- `frontend/src/pages/Dashboard.tsx` — **two** horizontal-scroll strips of `DetectionCard`s (the `SystemMonitor` "Station" card moved to the Hardware page): **Predictions Today** (everything at/after local midnight, up to `TODAY_LIMIT=100`) and **Recent Predictions** (the latest `RECENT_LIMIT=16` detections from *before* today). The two queries are disjoint around local midnight via the API's `from`/`to` timestamp filters: today uses `from = midnight`, recent uses `to = midnight − 1s`. Bounds are sent as **naive local ISO strings** (no `Z`/offset) via `toNaiveISO` because detection timestamps are written with `datetime.now()` (naive local) and the API compares against them directly — a UTC offset would shift the day boundary. A single min-confidence slider (0–100%) drives both strips via the `min_confidence` query param (0 = show all); it tracks a live display value and only commits to the fetch on release (mouse up / touch end / key up), so dragging doesn't fire a request per intermediate value. Cards open the `Lightbox` species comparison panel on click; the Dashboard owns a `lightbox` `{section, index}` state so prev/next/delete navigate within the correct strip, reset when the strips reload or a detection is deleted (deletes remove the row from both lists)
- `frontend/src/pages/History.tsx` — full-page history view; filter bar (species dropdown + from/to date pickers + min-confidence slider mapped to the `min_confidence` query param, 0 = show all; the slider only commits to the fetch on release so dragging doesn't refetch per intermediate value), tab switcher (Timeline | Gallery), infinite-scroll pagination (20/page via `IntersectionObserver`); owns all filter/pagination/lightbox/selection state and passes it down to sub-views. Filtering is done server-side so pagination stays correct. When appending a page it **dedupes by `id`** (keeps only rows not already held) before merging: the detector writes new detections live, so a row inserted between page fetches shifts the offset window and makes the next page re-return rows already shown — without the dedupe those rows render twice in both Timeline and Gallery (which share this one list). The offset cursor still advances by the raw page length. `removeDetections(ids)` reaps already-deleted rows from local state (list + selection + pagination offset) and closes the lightbox — the actual API deletes happen in `Lightbox`/`FileDownloader`, this only keeps the UI in sync
- `frontend/src/pages/Camera.tsx` — Camera tab with two sub-tabs. **Test**: a "Test Camera" button fetches `/api/camera/snapshot` as a blob (so HTTP errors surface as a message, not a broken image), cache-busts per click, and displays the returned frame; revokes object URLs on replace/unmount. **Detection region**: renders `CropEditor`
- `frontend/src/components/CropEditor.tsx` — interactive detection-region editor; loads a full-sensor preview (`/api/camera/snapshot/full`) rendered at the true sensor aspect ratio (`object-fit: fill` into an aspect-ratio container) and overlays a draggable/resizable box (corner handles, `pointer` events) tracked in normalized `[0,1]` coords over the displayed preview — exactly the space `POST /api/camera/crop` expects. "Apply region" persists + applies it live; "Reset to default" restores the feeder crop; "Reload preview" refetches
- `frontend/src/components/Timeline.tsx` — chronological paginated list of `DetectionCard`s with an `IntersectionObserver` sentinel for infinite scroll; opens lightbox on thumbnail click; full-res images are never loaded until the lightbox is opened
- `frontend/src/components/Gallery.tsx` — uniform thumbnail grid with checkbox-based multi-select (checkbox overlay + ring); `IntersectionObserver` for infinite scroll; integrates `FileDownloader` toolbar; opens lightbox on thumbnail click
- `frontend/src/components/Lightbox.tsx` — full-screen **lightbox** opened on thumbnail click (Gallery, Timeline, and Dashboard) — an in-app overlay, never a new page. Shows the captured full-res image with the species / both-confidences (`% match` for classification + `% spotted` for the YOLO detection score, the latter omitted when null) / time-ago caption, a download link, and a **Delete** action (confirm → `api.detections.delete` → `onDelete(id)`). The species **reference panel is hidden by default** and opened via a vertical **"Reference" tab** on the right edge of the image (toggle; emerald when open). When opened, the panel (`ReferencePane`) is locked to the image's **exact rendered pixel size** — measured live via a `ResizeObserver` on the `<img>` (re-measured on image swap / viewport resize) and applied as inline `width`/`height` — so it always matches the image and **never offsets it**; overflowing reference content scrolls internally (`overflow-y-auto`) instead of growing the panel. The panel **unfolds/folds with a 300ms transition** rather than snapping: it stays mounted once `imgSize` is known, and an outer `overflow-hidden` wrapper animates `width` (0 ↔ `imgSize.w`), `margin-left` (0 ↔ the gap to the image) and `opacity`, clipping the fixed-size inner card so its content never reflows mid-animation; the image is centred in the row so it glides aside as the panel grows (the global `prefers-reduced-motion` guard in `index.css` zeroes the duration, so it snaps for users who ask for that). The wrapper is `aria-hidden` + `pointer-events-none` while closed. The image itself is capped (`max-h-[80vh] max-w-[44vw]`, no `object-contain` letterbox so the element box equals the visible image) to leave room for the side-by-side panel. A **detection-box overlay** is drawn on the image from the persisted normalized `box_x/y/w/h` (an absolutely-positioned `pointer-events-none` `<div>` sized in `%` over the image, so it scales with the capped render). A **"Box on/off" toggle** in the caption bar shows/hides it; it is **on by default** and the toggle + overlay are hidden entirely for legacy rows whose box fields are null. A **Photo | Video segmented toggle** is **always shown** in the caption bar: when the detection has a `video_path`, Video mode swaps the `<img>` for a `<video controls autoplay loop muted poster={thumbnail}>` from `videoUrl`, and the box overlay/toggle (meaningful only on the still) are hidden while in Video mode; `mode` resets to Photo whenever the detection changes (so navigating to a clip-less sighting never lands on a blank player), and the Download link points at the clip in Video mode. When the detection has **no** clip, the Video button stays visible but is **disabled** (greyed, `cursor-not-allowed`) and, on hover, shows a `title` tooltip explaining why — mapped from `no_video_reason` by `noVideoReasonText()` ("recorder was busy with another sighting" / "video recording is turned off" / a generic fallback). It uses `aria-disabled` rather than the native `disabled` attribute so the hover tooltip still fires (browsers suppress hover events on natively-disabled buttons), and guards its `onClick`. `ReferencePane` fetches `api.species.reference(detection.species)` and shows the reference image(s) (prominent image + thumbnail strip when there are several), scientific name, summary, behaviour, per-image attribution/license, and a Wikipedia link (the only external nav). Refetches whenever prev/next changes the species (stale-response guard); graceful states for loading, 404 ("No reference available for this species yet"), and errors. `Timeline`/`Gallery`/`History`/`Dashboard` pass `onDelete` through. Keeps Esc/←/→ keyboard nav, prev/next arrows, body-scroll lock, and backdrop-click-to-close
- `frontend/src/components/FileDownloader.tsx` — manages bulk-select toolbar (select all / clear / count); streams ZIP download via `fetch()` + `ReadableStream`; tracks progress from `Content-Length` header; triggers browser download via `URL.createObjectURL`. Also renders a bulk **Delete** button: confirms, deletes the selected ids one-by-one via `api.detections.delete`, and reports the successfully-deleted ids to `onDeleteSelected` (partial failures still remove what succeeded)
- Build: `npm run build` (from `frontend/`) outputs to `frontend/dist/`; served by FastAPI at `/` via `StaticFiles`
- Dev: `npm run dev` proxies `/api/*` to `http://localhost:8080`
- `Dockerfile.api` — multi-stage image: Node 20 builds the frontend, Python 3.11 runs the API; exposes port 8080

**`birdscanner/detector/camera_server.py`** — detector control HTTP server (camera snapshots + crop control + detection deletion):
- `encode_jpeg(frame)` / `capture_jpeg(picam2)` — encode an RGB frame (RGB→BGR for OpenCV; the stream is `BGR888`, see "Colour channel order") / capture one from the `main` stream and encode it
- `start_camera_server(picam2, port, crop_controller=None, delete_detection=None)` — starts a `_ControlServer` (a `ThreadingHTTPServer` subclass that carries the `picam2`/`crop_controller`/`delete_detection` deps) on a background daemon thread. The request handler is a **module-level** `CameraRequestHandler` that reads those deps off `self.server` — not a per-server closure (that nesting made the factory trip `too-many-statements`). Routes: `GET /capture` (fresh JPEG of the cropped feed); when a `crop_controller` is supplied, `GET /capture/full` (full-sensor preview), `GET /crop` (current region as JSON), `POST /crop` (apply `{nx,ny,nw,nh}` or `{reset:true}`); and when a `delete_detection` callback is supplied, `DELETE /detections/{id}` (returns 204/404/500). With no controller the crop routes 404 (legacy snapshot-only mode); with no delete callback the delete route 404s. `camera_server_port()` reads `CAMERA_SERVER_PORT` (default 8000)
- Wired into `main.py` after `picam2.start(...)` so the read-only API can surface a live "Test Camera" image, the crop editor, **and** delete detections even though the detector owns the camera + read-write data volume exclusively; the delete callback is bound to the same engine the `DetectionWriter` uses (see `birdscanner/db/deleter.py`). Shut down on `KeyboardInterrupt`. NB: because the server starts only after the camera initialises, deletes (like snapshots/crop) are unavailable while the camera is down

**`birdscanner/detector/crop.py`** — pure, camera-independent crop domain (fully unit-tested in `tests/test_crop.py`):
- `CropRegion` — `(x, y, w, h)` in **unflipped raw sensor pixels**; `.clamped()` enforces `MIN_CROP_PX` and sensor bounds (`SENSOR_W=4056`, `SENSOR_H=3040`)
- `NormalizedBox` (`nx,ny,nw,nh`) and `SensorDimensions` (`w,h`) — small `NamedTuple` value objects that carry the UI box and sensor dims so the conversion functions (and `CropController`) don't take four/six scalar args
- `default_crop_region()` — the historical 900×900 feeder crop
- `normalized_to_sensor(box, sensor=SensorDimensions())` / `sensor_to_normalized(region, sensor=SensorDimensions())` — convert between a `NormalizedBox` (fractions over the displayed preview) and sensor coords via a **direct per-axis scale, no rotation**: libcamera applies `ScalerCrop` in the same orientation as the (vflip+hflip) transformed preview, so the displayed box maps straight to the matching sensor region (verified empirically — inverting the rotation cropped the diagonally-opposite corner)
- `main_stream_size_for_crop()` — the `main` ISP output size matching a crop's aspect ratio (longer edge `DEFAULT_LONG_SIDE=640`, even-aligned), so a non-square crop is not stretched
- `load_crop_region()` / `save_crop_region()` (atomic) — JSON persistence at `crop_config_path()` (env `CROP_CONFIG_PATH`, default `/data/crop.json`); any read error falls back to the default so a corrupt file never blocks startup

**`birdscanner/detector/crop_controller.py`** — `CropController(picam2, config)` applies + persists the crop on the running camera, serialising all camera access behind `camera_lock` (an `RLock` shared with `main.py`'s capture loop). Its static configuration (starting `region`, `main_size`, `config_factory`, `config_path`, and `sensor` dims) is bundled into a `CropControllerConfig` dataclass so the constructor takes just the live `picam2` handle + that config:
- `set_from_normalized()` / `reset_to_default()` → `_apply()`: a **pan/zoom at the same aspect ratio** applies live via `set_controls({"ScalerCrop": ...})`; an **aspect-ratio change** triggers a `stop()`→`configure()`→`start()` reconfigure (using the `config_factory` from `main.py`) to resize the `main` stream
- `capture_full_preview_array()` — briefly widens `ScalerCrop` to the full sensor under the lock, pulls a settled frame (`_capture_settled` waits for the control to take effect), then restores the previous crop; the momentary widening is a deliberate, brief glitch in the live feed while configuring
- `get_state()` — returns the crop as sensor pixels + normalized box + sensor dimensions (for the UI)

**`birdscanner/detector/track_logging.py`** — the `tracking` logger module: `configure_logging(debug)` sets up the stdout stream handler at DEBUG/INFO (called once from `main`), and `TrackingLogger` logs stable-track and track-deletion events (used by `gating.build_gating`). These event lines are deliberately terse — just `track_id`/`stable_frames` (+ `missing_frames` on delete); the box is **not** logged (noise) and the species is **not** logged here (it is `None` at these points anyway). The species is instead logged once per saved detection by `classification_pipeline._persist_detection` as a `Bird classified: track_id=… species=… confidence=…%` line followed by a `Saved to <path>` line, so a species only appears in the log when a stable track is actually classified as a bird and persisted

### Model files (not in repo, must exist on the Pi)

Paths are resolved by `birdscanner/detector/paths.py` (env-overridable via `ASSETS_DIR` / `MODEL_DIR`).

| Purpose | Path (relative to repo root) |
|---|---|
| Object detection (IMX500 firmware) | `/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk` (YOLO11n; emits normalized, `xy`-ordered boxes — see `birdscanner/detector/config.py`) |
| Species classifier | `assets/models/convnext_v2_tiny_int8.onnx` |
| Class-to-index mapping | `assets/models/convnext_v2_tiny.onnx_class_to_idx.json` |
| COCO labels | `assets/labels/coco_labels.txt` |

### Bounding box format

All boxes throughout the codebase are `(x, y, w, h)` in ISP output pixel coordinates after `imx500.convert_inference_coords`. The `preprocess_roi` function expands the box to a square with 20% padding before passing to the classifier.

### Colour channel order (RGB vs BGR)

The `main` ISP stream is configured with `"format": "BGR888"` — **not** `"RGB888"`. picamera2's `888` format names are byte-reversed relative to the numpy array they produce: `"BGR888"` yields an `[R, G, B]`-ordered array, while `"RGB888"` yields `[B, G, R]`. Everything downstream assumes **RGB**: the ConvNeXt classifier (ImageNet-RGB normalisation in `build_preprocessing`), `save_thumbnail` (`PIL.Image.fromarray`), and the `cv2.cvtColor(..., RGB2BGR)` writes in `classification_pipeline.py` / `camera_server.py`. Requesting `"RGB888"` therefore swaps red↔blue in every saved image **and** feeds the classifier mis-ordered channels, producing garbage species predictions. If colours ever invert again, check this format string first.

## Conventions

- All functions must have type hints and docstrings.
- The codebase is one package, `birdscanner`, with absolute imports; the layering `detector → ml → db` / `api → db` is one-way (`ml/` never imports `detector/` or `api/`).
- Runtime code lives under `birdscanner/`; dev-only scripts (never imported by the services, excluded from the Docker images) live under `tools/`.
- High-confidence classified bird images are written to `$IMAGE_DIR/{species}/` (env var, defaults to `/home/stefan/Pictures/bird_detections`). A 200×200 JPEG thumbnail is saved alongside each image with a `_thumb.jpg` suffix.
- `birdscanner/db/` tests use SQLAlchemy `StaticPool` to share an in-memory SQLite connection across threads (via the shared `engine` fixture in `tests/conftest.py`).
- `birdscanner/api/` tests (under ``tests/api/``) override FastAPI dependencies via ``app.dependency_overrides`` (the ``make_client``/``client`` fixtures in ``tests/api/conftest.py``) so no real DB or filesystem is needed.

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
- The `data` Docker volume is the single source of truth: the `detector` service writes images, the per-detection **mp4 clips** (alongside the stills under `$IMAGE_DIR/{species}/`), the SQLite DB, and the crop-region JSON (`CROP_CONFIG_PATH`, default `/data/crop.json`); the `api` service mounts it read-only. The crop config flows the other way too — the API never writes the file; it `POST`s crop changes to the detector's HTTP server, which applies them live and persists the JSON.
- **Video encoding is software** (the Pi 5 has no hardware encoder) and uses the apt `python3-opencv` FFmpeg videoio backend for `mp4v` — already pulled in by `python3-opencv` in `Dockerfile.detector`, so no extra package is needed. If clips fail to write (empty/absent mp4s), check that the opencv build has videoio/FFmpeg support.
- The **species-reference bank** also rides on the data volume at `/data/species_reference` (`SPECIES_REFERENCE_DIR`). It is built offline (`tools/build_species_reference.py`) and **not** baked into the api image — copy `manifest.json` + `images/` into `<data-volume>/species_reference/` on the host. Neither service writes it; the api just reads it (read-only mount). When absent, the Lightbox reference panel 404s for every species ("No reference available").
- `privileged: true` is scoped to `detector` only (required for IMX500 camera device access).
- The `detector` `expose`s port 8000 (the control server from `birdscanner/detector/camera_server.py`) on the internal compose network only — it is **not** published to the host. The `api` reaches it via `DETECTOR_URL` (default `http://detector:8000`); the detector binds the port from `CAMERA_SERVER_PORT` (default 8000). Both vars live in `.env`/`.env.example` and the defaults work as-is — only change them together. This powers the Camera tab's "Test Camera" button, the detection-region editor (`/crop`, `/capture/full`), **and** detection deletes (`DELETE /api/detections/{id}` proxies to `${DETECTOR_URL}/detections/{id}`).
- The `detector` image (`Dockerfile.detector`) is based on `dtcooper/raspberrypi-os:bookworm` so the system `python3` can import the apt-installed `python3-picamera2` / `python3-libcamera` bindings (these are built natively against the Pi's libcamera and are **not** on PyPI). `numpy`, `opencv` and `pillow` are installed via apt (`python3-numpy` / `python3-opencv` / `python3-pil`); the apt `numpy` is `1.24.2`, which is what the apt-built picamera2/simplejpeg/opencv stack is compiled against. `requirements.detector.txt` therefore pins `numpy==1.24.2` (so pip does not pull numpy 2.x into `/usr/local` and shadow the apt copy — that crashes the detector with `numpy.dtype size changed`) and `onnxruntime==1.23.2` (the version known to run against numpy 1.24.2 on this Pi). These plus `sqlmodel` are pip-installed into the system interpreter with `--break-system-packages`. The IMX500 firmware + `.rpk` network models (`/usr/share/imx500-models/...`) come from the `imx500-all` apt package baked into the image. A plain `python:3.x` base will crash the detector with `ModuleNotFoundError: No module named 'libcamera'`.
- The `detector` service additionally mounts `/run/udev:ro` (libcamera enumerates cameras via udev), the `/dev/dma_heap` device (picamera2 buffer allocation), `/sys/kernel/debug` (debugfs), and `/usr/share/imx500-models:ro` — all required for the camera to initialise inside the container. The `imx500-models` mount exposes the **host's** `.rpk` network firmware to the container: the image's `imx500-all` apt package does not ship the `imx500_network_yolo11n_pp.rpk` model the detector loads (see `birdscanner/detector/config.py`), so without the mount the detector logs `Camera not available (Firmware file ...yolo11n_pp.rpk does not exist.)` and retry-loops forever. Mounting the host directory (which has the model, since the bare detector runs there) keeps the container in sync with the host's apt package without baking the binary into the image. The debugfs mount is needed because `IMX500.__init__` opens `/sys/kernel/debug/imx500-fw:<id>/fw_progress` to track on-chip firmware upload; Docker does not expose debugfs to containers (even privileged ones) by default, so without this mount the detector crashes with `FileNotFoundError: ... /sys/kernel/debug/imx500-fw:11-001a/fw_progress`.
- `main.py` configures the camera with `buffer_count=6` (not the bare-Pi example's `12`). Inside the container the kernel CMA / dma-heap pool is host-global (Docker memory limits don't apply to it) and is shared with the IMX500 firmware upload and the always-allocated 2028×1520 raw sensor stream (~3 MB/buffer, the dominant consumer — it's fixed at the sensor's smallest mode and allocated by the Pi 5 PiSP pipeline regardless of config, so it can't be shrunk). With 12 buffers this exhausted CMA and crashed `picam2.start()` with `OSError: [Errno 12] Cannot allocate memory` at the dma-heap `alloc` ioctl. 6 buffers halves DMA pressure while keeping jitter margin; the `main` stream is sized so its longer edge is 640 (square for a square crop, smaller on the short side for a non-square crop — see `main_stream_size_for_crop`), so it never exceeds the old 640×640 footprint. The ConvNeXt classifier crops bird ROIs from `main` (the on-chip detector runs at a fixed 320×320 baked into the `.rpk`, independent of the streams). If 6 still ENOMEMs, increase the host CMA pool in `/boot/firmware/config.txt` rather than dropping `main` resolution.
- Model files (`assets/models/convnext_v2_tiny_int8.onnx`) must be present on the Pi; they are not included in the image.
- The detector runs as `python3 -m birdscanner.detector.main` from `WORKDIR /app`. It loads its data files **package-relative** via `birdscanner/detector/paths.py` (anchored at the package location, resolving to `/app/assets/...` — no longer cwd-relative), so the Dockerfile copies the code to `/app/birdscanner/` and the assets to `/app/assets/` (labels under `assets/labels/`, the classifier ONNX + its class map under `assets/models/`). Only the single `convnext_v2_tiny_int8.onnx` is copied from `assets/models/` — any other model files there are training/conversion scratch. If the model is missing, onnxruntime fails at startup with `NO_SUCHFILE: Load model from .../convnext_v2_tiny_int8.onnx failed`.
- If the camera is unavailable (dev-node missing — unplugged, mis-seated, or device access not granted), the `detector` does **not** crash-loop: `wait_for_camera()` logs `Camera not available (...). Retrying in 30s...` and retries until the camera appears. The `api` service is independent and continues serving the site; because the detector initialises the DB schema on startup, the UI loads (showing existing images, or an empty gallery on a fresh volume).
