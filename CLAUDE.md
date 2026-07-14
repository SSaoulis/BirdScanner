# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BirdScanner is a real-time bird detection and classification system for a Raspberry Pi with the Sony IMX500 AI Camera. The IMX500 runs object detection (YOLO11n) on-chip; detections are passed to a ConvNeXt V2 Tiny ONNX model for bird species classification. The system must run on the Pi — `picamera2` and `libcamera` are not available on other platforms.

## How to develop

The development cycle is based on a worktree approach. Therefore there are a fixed set of rules to apply to develop: 

- Start by creating a worktree from main.
- ALWAYS pull latest changes from main onto the worktree before any files are read/analysed.
- Once a change is made, ALWAYS update the CLAUDE.md, and make a PR (not a draft PR).
- Generate a sub-agent to review the changes. If the changes are good, allow the agent to accept the PR, then merge it into main.


Use the .venv in the project root - this contains the linting/formatting tools and all required packages installed. 

## Commands

### Run the app (on Raspberry Pi only)
```bash
python -m birdscanner.detector.main   # from the repo root
```
Runtime behaviour is set in `birdscanner/detector/config/config.py` (not CLI flags); edit the module-level `config` instance. Key fields: `multithread` (bg classification thread, default `True`), `object_duration_threshold` (seconds a track must be stable before classifying; `<= 0` → one stable frame; default `0.2`), `debug`, `preview`, `save_video` + `video_pre_roll_seconds`/`video_post_roll_seconds`, plus `model`, `fps`, `threshold`, and the IMX500 intrinsics overrides.

### Run the app OFF the Pi (emulated camera, for local dev)
```bash
# needs assets/models/yolo11n.onnx (dev-only, out-of-band — see requirements.dev.txt)
FAKE_CAMERA_SOURCE=some_feeder_clip.mp4 python -m dev.run_emulated
python -m dev.run_emulated   # no source → cycles the bundled tests/_test_images stills
```
Installs fake `picamera2`/`libcamera` modules (backed by a real YOLO11n ONNX detector) and runs the real `main()`, so the whole detector (classifier, geomodel prior, SQLite writer, control server, clip encoding) runs locally. Point the API at the same `DB_PATH`/`IMAGE_DIR` to watch detections in the UI. See `dev/emulation`.

### Tests
```bash
pytest tests/                                                    # all
pytest tests/ml/test_tracking.py::test_update_tracks_for_frame_increments_stability  # single
pytest --cov=birdscanner tests/                                  # coverage
```
`pyproject.toml` sets `pythonpath = ["."]`, so plain `pytest tests/` resolves the package with no editable install. Tests needing ONNX model files skip automatically when the model is absent.

`tests/` mirrors the `birdscanner/` package (`tests/ml/`, `tests/db/`, `tests/api/`, `tests/detector/`, `tests/tools/`); each dir is a package so duplicate basenames are safe. Shared fixtures live in the nearest `conftest.py`:
- **`tests/conftest.py`** (global) — `frame_factory` + the in-memory DB fixtures (`engine`/`session_factory`/`image_dir`/`detection_factory`), global because both `db` and `api` suites need them.
- **`tests/api/conftest.py`** — `make_client`/`client` (deps overridden), `seeded_detections`, `FakeHttpxResponse`.
- **`tests/ml/conftest.py`** — injectable pipeline fakes plus the real-model fixtures `bird_image_cases` and `real_classifier` (both skip when assets absent, module-scoped).

The **off-Pi emulator** has tests under `tests/detector/`: `test_capture_loop.py` runs the real `build_camera` + `_run_capture_loop` off-Pi with a stub detector (covers `camera.py`/`main.py`'s loop), plus `test_fake_camera.py`, `test_emulation_yolo.py`, `test_emulation_frames.py`. **End-to-end tests** use `tests/_test_images/` (committed labelled JPEGs + `bounding_box_locations.json`): the hand-labelled boxes stand in for the IMX500 `.rpk` (Pi-only); everything downstream runs for real (`tests/ml/test_bird_species_classification.py`, `tests/ml/test_end_to_end_detection.py`).

### Type checking, linting, formatting
```bash
mypy birdscanner tools dev tests --check-untyped-defs   # single pass (one package, absolute imports)
black birdscanner tools dev tests                       # run black FIRST
pylint birdscanner tools dev                            # source (root .pylintrc)
pylint tests --rcfile=tests/.pylintrc                   # tests (relaxes pytest idioms)
```
`mypy.ini` / `.pylintrc` silence the Pi-only native bindings (`libcamera`, `picamera2`) and untyped libs (`onnxruntime`, `psutil`, `paramiko`, `PIL`, `cv2`, `onnx`, `matplotlib`). Pylint is kept at **10.00/10**. The complexity/argument checks (`too-many-*`, `duplicate-code`) are **deliberately enabled** — do not add `# pylint: disable=too-many-*`; instead collapse wide signatures into **parameter/value objects** (e.g. `ml/detection_utils.Box`, `classification_pipeline.PipelineContext`, `crop.NormalizedBox`, `detections.DetectionFilters`) or **decompose** the function. `generated-members=cv2.*` (OpenCV members are dynamic).

## Architecture

### Detection pipeline (frame loop)

```
IMX500 on-chip inference
  └─ parse_detections()                              # extract Detection objects from inference tensor
       └─ StableDetectionTracker.update_frame()      # IoU-based multi-frame tracking
            └─ process_detections() [picam2 pre_callback]
                 └─ ClassificationManager.process()
                      └─ process_single_detection_with_stable_tracks()
                           └─ BestFrameSelector.take(track_id)  # best frame across the track
                           └─ Classifier.classify()             # ConvNeXt V2 Tiny ONNX
                                └─ GeoPriorAdjuster.adjust()     # optional Bayesian reweight by location
                                └─ save clean full image + 200×200 thumbnail to IMAGE_DIR/{species}/
                                └─ VideoRecorder.trigger()       # short mp4 clip
                                └─ DetectionWriter.write()       # SQLite detections table
```

The still + thumbnail come from the **best frame observed across the track** (highest YOLO confidence), not the trigger frame (`ml/best_frame.py`). The saved still is the **raw frame — no box is drawn on it**; the box is persisted as normalized `[0,1]` coords (`box_x/y/w/h`) so the UI can overlay it with a toggle. The clip finishes encoding a few seconds after the DB row is written, so `/video` briefly 404s while the UI shows the still.

**Video clip source** (`config.video.full_fov`, default `False`): by default the clip records the same cropped, ISP-processed `main` frames as the still (matching crop/aspect/quality, browser-playable H.264). Set `full_fov = True` to record the whole uncropped FOV instead — fed from the always-allocated **raw** sensor stream (there is only one `ScalerCrop` per camera, applied to every processed stream), debayered by `detector/hardware/raw_frame.py`. Full-FOV is ISP-less (softer, 4:3) and costs extra CMA, so it is off by default.

### Package layout

One package, `birdscanner/`, run via `python -m ...` with absolute imports. Four layered subpackages plus assets + dev tooling:

```
birdscanner/
  detector/   Pi-only camera + hardware control and the entry point:
                main.py, paths.py          entry point + data-file resolution
                hardware/                  camera, raw_frame, crop, crop_controller,
                                           video_recorder, camera_server
                pipeline/                  gating, track_logging, geo_priors
                config/                    config, settings, settings_controller
  ml/         platform-independent inference (object_detection, detection_utils, tracking,
              classification, classification_pipeline, best_frame, geomodel)
  api/        FastAPI REST API + routers
  db/         SQLite persistence
assets/       labels/, models/ (classifier ONNX + class map), species_reference/
dev/          off-Pi camera emulator (run_emulated + emulation/); may import birdscanner
tools/        offline builders/scripts — NOT runtime
```

**Import direction is one-way:** `detector → ml → db` and `api → db`. `ml/` must never import from `detector/` or `api/`. `birdscanner` must never import `dev/` or `tools/`.

### Key modules

**`ml/object_detection.py`** — `Detection` (box + category + confidence), `parse_detections(metadata, context, threshold, inference_roi=None)` (extract from IMX500 tensor; `context` is the `InferenceContext` value object bundling `imx500`/`intrinsics`/`picam2`; when `inference_roi` — an `InferenceRoi` carrying the ROI + sensor dims — is given, remaps each box from ROI-relative to full-sensor fractions before `convert_inference_coords`, see the on-sensor-inference Gotcha), `get_labels`, `filter_included_detections(detections, labels, included)` (keeps only object-detection classes in the include set/allowlist, matched case-insensitively; called per-frame in `_run_capture_loop` so non-included classes e.g. `"bench"`/`"person"` never reach the tracker — by default only `"bird"` is kept; an **empty** set keeps everything, the no-op fast path). `last_detections` module-global fallback.

**`ml/detection_utils.py`** — stateless geometry/drawing helpers over the `Box` value object (owns `.iou`/`.padded_square`/`.crop`/`.normalized`): `iou`, `preprocess_roi` (expands box to a padded square then crops), `draw_boxes` (live-preview annotation only — no longer drawn on the saved image), `normalized_box` (pixel box → `[0,1]` fractions), `save_thumbnail` (200×200 JPEG), `label_for_category` (bounds-checked label lookup returning `None` out of range — the IMX500 SSD model occasionally emits an out-of-range category, so all `labels[category]` access goes through this).

**`ml/tracking.py`** — `StableDetectionTracker` (IoU tracker; a detection must match across `min_stable_frames` consecutive frames before it classifies; each track classified at most once via `mark_classified`), the pure primitives `StableTrack`/`match_detection_to_track`/`update_tracks_for_frame`/`should_classify_track`, and the `stable_detection_tracker` module default.

**`ml/classification_pipeline.py`** — classification orchestration + persistence:
- `PipelineContext` — parameter object bundling every per-detection dependency (`classifier`, `tracker`, `classify_fn`, `detection_writer`, `best_frame_selector`, `record_fn`, `video_frame_fn`, optional `video_frame_source` and `geo_adjuster`) plus two live-editable save-side settings (`save_confidence_threshold` default 0.4, `ignore_species` lower-cased set). These live here (not on `config`) because `ml/` must not import `detector/`; the Settings page mutates them via `settings_controller`. `Still`/`Classification` are small value objects. `_predict_species` computes the classifier softmax once, always records `classifier_scores` (top-K), and when a `geo_adjuster` is wired runs the geomodel Bayesian update so `species`/`confidence` are the geomodel-corrected posterior (`classifier_species`/`classifier_confidence` keep the pre-adjustment pick, `geo_scores` the pre-normalised top-K). `_should_persist` gates on the posterior confidence **and** the ignore list.
- `ClassificationManager(context, *, use_multithreading=False, queue_maxsize=0)` — sync/async (threaded `Queue`) dispatch. **The queue must be bounded** (each item pins a ~5 MB `main` frame); `build_manager` passes `CLASSIFICATION_QUEUE_MAXSIZE = 32` so a busy feeder drops excess frames rather than leaking memory (safe: `BestFrameSelector` retains each track's best frame independently). Both paths run through `_dispatch`, which catches/logs any per-detection exception so one bad detection can't kill the worker.
- `process_detections` — picam2 `pre_callback`: draws boxes, feeds every clean frame to the pre-roll buffer (full uncropped FOV when `video_frame_source` is wired, else cropped `main`), offers each bird detection to `best_frame_selector`, queues bird detections.
- `process_single_detection_with_stable_tracks(item, context)` — the only classification path; decomposed into `_best_still`/`_classify_track`/`_save_still_and_thumbnail`/`_start_clip`/`_persist_detection`. Uses the track's best frame for the ROI, saved still, thumbnail and persisted box; `_start_clip` returns `(video_path, no_video_reason)` (`"recorder_busy"`/`"disabled"`). **Skips a zero-area box** whose ROI is empty (would crash PIL); the track stays unclassified for a later frame.
- `setup_classifier`, `run_bird_classification`, `IMAGE_DIR` (from the `IMAGE_DIR` env var; default `/home/stefan/Pictures/bird_detections`).

**`ml/best_frame.py`** — `BestFrameSelector` (thread-safe `observe`/`take`/`discard`; keeps the single highest-YOLO-confidence `(frame, box)` per track_id, freed on track end via the tracker's `on_track_deleted` callback). `BestCandidate` is the retained dataclass.

**`ml/geomodel.py`** — BirdNET geomodel (spatio-temporal prior), the geomodel↔classifier label crosswalk, and the runtime Bayesian reweight:
- Runtime: `week_of_year` (bins a date into `1..48`), `build_prior_matrix` (aligns the `{species: [48 floats]}` store to a `(48, n_classes)` matrix; absent species floored not zeroed, unmapped classes get a neutral `1.0`), `geomodel_posterior` (returns posterior + pre-normalised product; falls back to classifier probs if the product is 0), `GeoPriorAdjuster(priors, idx_to_class).adjust(probs, week)` → `GeoAdjustment`. Built by `pipeline/gating.build_geo_adjuster` from the DB prior.
- Startup projection: `load_name_mapping`/`project_to_classifier`/`compute_classifier_priors` project the geomodel prior onto the classifier's ~706 classes (`~706 species × 48 weeks`, persisted to the DB on startup).
- The crosswalk (`normalize_common_name`, `build_name_mapping`) reconciles the geomodel's eBird/Clements names with the classifier's IOC-style names; it is **classifier-keyed** because the relation is many-to-one (several classifier classes can share one geomodel species after an eBird lump). The curated result is `assets/labels/geomodel_classifier_map.json` (tracked source; 706/707 classes mapped, only the `Unknown` sentinel left). Rebuild with `tools/build_geomodel_map.py`.

**`ml/classification.py`** — `ONNXClassifier` (raw wrapper, `(N,C,H,W)` float32), `Classifier` (adds preprocessing + class mapping; `classify()` → `(species, confidence)` argmax, `predict_proba()` → full softmax for the geomodel update), `build_preprocessing` (PIL+NumPy resize→center-crop→ImageNet-normalize→NCHW), `top_k_predictions`.

**`detector/config/config.py`** — `Config` dataclass + `config` instance; edit values here. Nested sub-configs keep it small: `config.intrinsics` (`IntrinsicsConfig`: IMX500 intrinsics overrides) and `config.video` (`VideoConfig`: `save`, `pre_roll_seconds`/`post_roll_seconds`, `full_fov`). Top-level: `model`, `threshold`, `multithread`, `object_duration_threshold`, `restrict_inference_to_crop` (bool, default `True` — crop the on-chip DNN input to the detection crop region so the detector sees the same zoomed bird as the classifier; `False` reverts to full-FOV inference), `included_classes` (object-detection/YOLO-class allowlist, `set[str]` default `{"bird"}`, empty = keep all, read live), `debug`, `preview`, `latitude`/`longitude` (`Optional[float]`, default `None` — the geomodel location). `main.py` imports the instance as `app_config`.

**`detector/config/settings.py`** — user-editable runtime settings layered as a JSON overlay on `config` defaults (the Settings page). `Settings` dataclass of editable values (`detection_threshold`, `classification_threshold`, `ignore_species`, `included_classes` (allowlist, default `{"bird"}`, empty = keep all), `stability_seconds`, `image_dir`, video fields, `multithread`, `debug`, `latitude`/`longitude`). `LIVE_FIELDS`/`RESTART_FIELDS` partition which take effect immediately vs. at next boot (drives the UI "restart" badge; `latitude`/`longitude` are restart fields — the prior is only rebuilt at boot). `merge_settings` validates a partial update (bad key/value → `ValueError` without mutating). `load_settings`/`save_settings` (atomic) persist at `settings_config_path()` (env `SETTINGS_PATH`, default `/data/settings.json`).

**`detector/config/settings_controller.py`** — applies settings to the running detector. `apply_settings_to_config` runs once at startup (restart-only fields shape the build); `apply_settings_to_context` applies save-side fields to the `PipelineContext` (they live there because `ml/` can't import `detector/`). `SettingsController.update(updates)` validates + persists + live-applies (mutates `config.threshold`, `config.included_classes`, the context save-fields, the `tracking` log level) and returns `get_state()` (`needs_restart` true when a restart-only field changed).

**`detector/paths.py`** — package-anchored data-file resolution (relative to the package, not cwd, so the detector runs from anywhere). `assets_dir()`/`model_dir()` overridable via `ASSETS_DIR`/`MODEL_DIR`; helpers for the COCO labels, class map, classifier model, and the geomodel model/labels/map.

**`detector/main.py`** — entry point + capture loop only. `main()` is a short startup script: `load_settings`/`apply_settings_to_config` → `configure_logging` → `make_engine`/`init_db` → `refresh_geo_priors` → `wait_for_camera`/`prepare_intrinsics` → `setup_classifier` → `build_gating` → `DetectionWriter`/`build_geo_adjuster`/`build_manager`/`apply_settings_to_context`/`SettingsController` → `build_camera` → optionally wire `video_frame_source` (only when `config.video.full_fov`) → `_start_control_server` → `_run_capture_loop`. `init_db` runs before camera init so the read-only API can serve even if the camera never comes up. `_run_capture_loop` holds the `CropController`'s `camera_lock` across `capture_metadata()`+`parse_detections()` (passing `camera.inference_roi_state.roi` so ROI-relative boxes are remapped to full-sensor coords), then filters via `filter_included_detections(results, labels, app_config.included_classes)` before storing (keeps only the allowlist, default `{"bird"}`; read live; empty = keep all). `_schedule_restart` does a delayed `os._exit(0)` so Docker's `restart: unless-stopped` relaunches the detector to pick up restart-only settings.

**`detector/hardware/camera.py`** — Pi-only camera bring-up (imports `libcamera`/`picamera2`; no test imports it). `wait_for_camera()` retries with backoff (never crash-loops). `prepare_intrinsics(imx500)` validates the object-detection task + applies intrinsics overrides. `build_camera(imx500, intrinsics)` loads the persisted crop, sizes the `main` stream to the crop aspect, applies `vflip=True, hflip=True` (camera mounted upside-down), starts the camera, restricts the **on-chip DNN input** to the crop region via `imx500.set_inference_roi_abs(inference_roi_for_crop(region))` (gated by `config.restrict_inference_to_crop`; wired as the `CropController`'s `set_inference_roi` callback so it re-arms on every live crop change) **and publishes that exact ROI into `Camera.inference_roi_state` (an `InferenceRoiState` holder) so `_run_capture_loop` can remap the network's ROI-relative output boxes** (see the on-sensor-inference Gotcha), builds the `CropController`, and — only when `config.video.full_fov` — requests the full-FOV unpacked raw stream via the **non-allocating** `generate_configuration([Raw])` (must **not** read `picam2.sensor_modes`, which allocates full-res buffers for every mode and exhausts CMA). Returns a `Camera` bundle.

**`detector/hardware/video_recorder.py`** — `VideoRecorder`: bounded `deque` pre-roll fed by `add_frame` (cheap, no encoding while idle — the Pi 5 has no hardware encoder, so all encoding is software/CPU). `trigger(dest_path)` snapshots pre-roll, collects `post_roll_seconds` live, then encodes to mp4 on a background thread. Encodes **H.264 (`avc1`)** — the only codec browser `<video>` can decode — falling back to `mp4v` (logged; won't play in-browser). **Single-flight**: a trigger while recording is declined (returns `False`).

**`detector/hardware/raw_frame.py`** — full-FOV clip frames from the raw sensor stream (only used when `config.video.full_fov`; pure cv2/numpy, runs off-Pi). `bayer_cv2_code` maps a libcamera raw format to the OpenCV demosaic code (letters swapped vs libcamera). `RawToRgb.convert` right-shifts to 8-bit, demosaics, downscales. `build_clip_frame_source(picam2)` reads the configured raw format back and returns a `request → RGB` callable (never raises; returns `None` on failure so the pipeline degrades to the cropped `main` frame). If clip colours ever invert, check the format→cv2 mapping here first.

**`detector/hardware/crop.py`** — pure crop domain (fully unit-tested). `CropRegion` `(x,y,w,h)` in unflipped raw sensor pixels (`.clamped()` enforces `MIN_CROP_PX`/`SENSOR_W=4056`/`SENSOR_H=3040`), `NormalizedBox`/`SensorDimensions` value objects, `default_crop_region()` (900×900 feeder crop), `normalized_to_sensor`/`sensor_to_normalized` (direct per-axis scale, no rotation — libcamera applies `ScalerCrop` in the transformed preview orientation), `main_stream_size_for_crop()` (the `main` ISP size matching the crop aspect, longer edge `DEFAULT_LONG_SIDE=1280`, even-aligned — sizes the still, classifier ROI and cropped clip), `inference_roi_for_crop(region)` (the `(l,t,w,h)` sensor ROI for `IMX500.set_inference_roi_abs`, a direct pass-through of the crop — the docstring carries the flip caveat if the DNN ends up looking at the diagonally-opposite region), `load_crop_region`/`save_crop_region` (atomic JSON at `crop_config_path()`, env `CROP_CONFIG_PATH`, default `/data/crop.json`).

**`detector/hardware/crop_controller.py`** — `CropController(picam2, config)` applies + persists the crop on the running camera, serialising camera access behind `camera_lock` (shared `RLock` with the capture loop). Static config bundled into `CropControllerConfig`. A same-aspect pan/zoom applies live via `set_controls`; an aspect change triggers `stop()`→`configure()`→`start()`. `capture_full_preview_array()` briefly widens `ScalerCrop` to the full sensor, pulls a settled frame, restores. `get_state()` returns crop as sensor pixels + normalized box.

**`detector/hardware/camera_server.py`** — detector control HTTP server (snapshots + crop + deletes + corrections + settings). `start_camera_server(picam2, port, deps=None)` starts a threading server on a daemon thread; deps collapse into `ControlServerDeps`. Routes (each enabled by its dep, else 404): `GET /capture`, `GET /capture/full`, `GET`/`POST /crop`, `DELETE /detections/{id}`, `PATCH /detections/{id}` (species correction — validates against `species_labels` ∪ custom species, `allow_new` registers a brand-new custom label; 400 JSON `{"error": …}` for an unknown label so the message survives the API proxy), `GET /labels` (sorted union), `GET`/`POST /settings`, `POST /restart`. Started only after the camera initialises, so these are unavailable while the camera is down. `camera_server_port()` reads `CAMERA_SERVER_PORT` (default 8000).

**`detector/pipeline/gating.py`** — classification-pipeline wiring (imports `ml/`/`config`/`db`, no picamera2, so importable without a camera). `build_gating(intrinsics)` builds the tracker (wired to a `TrackingLogger`), the `BestFrameSelector` (freed per-track via `on_track_deleted`), and the optional `VideoRecorder`; `min_stable_frames(fps)` converts `object_duration_threshold`×fps (floored at 1). `build_manager(classifier, gating, detection_writer, geo_adjuster=None)` assembles the `PipelineContext` + `ClassificationManager` (async with explicit `queue_maxsize=CLASSIFICATION_QUEUE_MAXSIZE=32`). `build_geo_adjuster(classifier, session_factory)` builds the runtime `GeoPriorAdjuster` from the DB prior; returns `None` (geomodel off) when there is no class map or no stored priors.

**`detector/pipeline/geo_priors.py`** — startup orchestration for the geomodel prior. `refresh_geo_priors(session_factory, lat, lon, *, compute=...)` runs once after `init_db`; **skips** when no location is set or the stored meta already matches (a cheap no-op on restart), else computes the `{species: [48 floats]}` prior and persists it. **Never crashes startup** (a missing model/inference error is logged and swallowed). `compute` is injectable so tests avoid the 28 MB ONNX.

**`detector/pipeline/track_logging.py`** — the `tracking` logger. `configure_logging(debug)` sets up the stdout handler (called once from `main`). `TrackingLogger(labels=None)` logs terse stable-track and track-deletion events (`track_id`/`stable_frames`, + `missing_frames` and `class=<COCO label>` on delete). The classified species is logged instead by `classification_pipeline._persist_detection` (`Bird classified: … / Saved to <path>`).

**`api/`** — FastAPI REST API. The DB is mounted **read-only** (`make_engine(read_only=True)`); the detector owns all writes, so write actions (delete, species correction, settings, crop) are **proxied** to the detector's control server via httpx (503 when unreachable). Modules:
- `main.py` — app factory; mounts routers; optionally serves `frontend/dist/` at `/` via `SPAStaticFiles` (a `StaticFiles` subclass that falls back to `index.html` on 404 so client-side deep links like `/history` work). API routes registered before the mount take precedence.
- `dependencies.py` — `get_session`/`get_image_dir`/`get_reference_dir` providers (env `DB_PATH`/`IMAGE_DIR`/`SPECIES_REFERENCE_DIR`).
- `detector_proxy.py` — `detector_error_detail(resp)`, the shared bit of proxy logic (pulls `{"error": …}` out of the detector's error body; centralised so pylint's `duplicate-code` stays quiet).
- `routers/detections.py` — `GET /api/detections` (paginated, filtered by `species`/`from`/`to`/`min_confidence` via the `DetectionFilters` `Depends`; ordered by timestamp desc then id desc so paginated pages don't overlap), `GET`/`DELETE`/`PATCH /api/detections/{id}` (delete + PATCH correction proxied to the detector; `allow_new` forwards to register a custom species).
- `routers/images.py` — thumbnail / full / video (mp4; 404 until encoded) / `download?ids=` (chunked ZIP).
- `routers/system.py` — `GET /api/system` (CPU/mem/disk/temp/uptime via psutil).
- `routers/network.py` — passive usage graph (a daemon sampler reads NIC counters every 3 s into a ~1 h ring buffer; `GET /api/network/history?range=`) + on-demand `POST /api/network/speedtest` (small Cloudflare up/down, never polled).
- `routers/species.py` — `GET /api/species` (counts), `GET /api/species/vocabulary` (proxies the detector's `GET /labels` — classifier classes ∪ custom species), `GET /api/species/expected?limit=` (the geomodel "expected this week" list, read from the DB prior for the current week; clean 200 with empty list when no location). The API image ships no `assets/`, so it must not import `ml.geomodel` — it uses a local `_current_geo_week()` copy of `week_of_year`.
- `routers/stats.py` — aggregate Statistics endpoints (prefix `/api/stats`), every one a read-only SQL `GROUP BY` (only buckets cross the wire): `summary`, `time-of-day`, `activity`, `timeline`, `daily-window`, `first-sightings`. A `StatsRange` `Depends` applies the optional `from`/`to`. (Gotcha: minute-of-day bucketing must use floor division `//` — SQLAlchemy `/` renders float division and splits buckets.)
- `routers/reference.py` — species reference/comparison data (prefix `/api/species`): `{name}/reference` (cached Wikipedia/Wikidata info; 404 when absent), `{name}/reference/images/{index}` + `.../thumbnail` (traversal-guarded, thumbnail falls back to full). Reads the manifest from `get_reference_dir()`; a missing manifest 404s cleanly.
- `routers/settings.py`, `routers/camera.py` — proxy the Settings page and the snapshot/crop endpoints to the detector.

**`db/`** — SQLite persistence. The detector owns all writes; the API mounts read-only.
- `models.py` — `DetectionRecord` (`detections` table). Two confidences: `confidence` (ConvNeXt classification) + nullable `detection_confidence` (YOLO). Nullable `box_x/y/w/h` (normalized box), `video_path` + `no_video_reason` (`"recorder_busy"`/`"disabled"`), the geomodel breadcrumbs `classifier_species`/`classifier_confidence`/`geo_scores`, `classifier_scores` (the classifier's own top-k softmax), and the correction fields `corrected`/`original_species`. Also `GeoPrior`/`GeoPriorMeta` (the prior tables) and `CustomSpecies` (user-added labels). All created by `init_db`'s `create_all`.
- `database.py` — `make_engine()` (`read_only=True` → `mode=ro` URI for the API) / `init_db()` / `make_session_factory()`; DB path from `DB_PATH`. `init_db` also runs `_migrate_detections_columns()` — an `ALTER TABLE ADD COLUMN` pass backfilling post-launch nullable columns (driven by `_DETECTIONS_ADDED_COLUMNS`; add new ones there).
- `writer.py` — `DetectionWriter`: fire-and-forget bg-thread writer; `write(record)` takes a pre-built `DetectionRecord`; `stop()` flushes.
- `deleter.py` / `corrector.py` / `custom_species.py` / `geo_prior_store.py` — synchronous, detector-owned stores (no camera/ml imports; run on the control server's HTTP thread). `delete_detection` (row + image/thumbnail/video, best-effort unlink), `correct_detection_species` (moves files to the new species folder, sets `corrected`/`original_species`, leaves model breadcrumbs), `list_custom_species`/`add_custom_species` (case-insensitive de-dup), and the geo-prior read/write (`read_meta`/`location_matches`/`has_geo_priors`/`replace_geo_priors`/`load_geo_priors`).

**`tools/`** (offline, not runtime) — `build_species_reference.py` (+ `reference_thumbnails.py`) builds the Wikipedia/Wikidata species-reference bank (manifest + images + thumbnails; incremental/idempotent; `--output-dir` targets the deployed data-volume bank, `--force`/`--limit`/`--throttle`; `overrides.json` is curated source). `build_geomodel_map.py` regenerates the geomodel↔classifier crosswalk (prints unmatched classes to curate; `--check` dry-runs). Plus `quantize`, `camera_smoke*`.

**`frontend/`** — React + Vite + Tailwind dashboard, themed as a naturalist's "field journal" (warm-paper palette, Fraunces + Hanken Grotesk). Tokens live in `frontend/tailwind.config.ts`; reuse them (and the `.eyebrow`/`.tnum` utilities in `index.css`) instead of raw Tailwind `slate-*`/`emerald-*`. The app is light-only. Key files:
- `api.ts` — typed fetch wrappers for all `/api/*` endpoints + the `Detection`/`SystemStatus`/`SpeciesSummary`/etc. interfaces and `timeAgo`/`formatUptime` helpers. `apiFetch` throws an `ApiError` carrying the HTTP status; `postJson`/`patchJson` parse FastAPI's `{"detail": …}` into the error message.
- `App.tsx` — `BrowserRouter` routes (`/`, `/history`, `/stats`, `/camera`, `/hardware`, `/settings`) + `<NavBar/>`. Statistics is `React.lazy`-imported (Nivo ~400 kB code-split).
- `components/NavBar.tsx` — sticky masthead; links inline on `lg`+, collapsed into a current-page dropdown below `lg` (from a single `NAV_ITEMS` array).
- `hooks/useMediaQuery.ts` — `useMediaQuery(query)` + `useIsDesktop()` (`lg` = 1024px), used by `Lightbox` to choose side-by-side vs stacked panels.
- Pages: `Dashboard.tsx` (`TodaySummary` day-totals panel under the title + ExpectedThisWeek band + FeaturedDetection hero + "Spotted today" strip + earlier-sightings grid grouped This week/This month/Earlier; local-midnight `from`/`to` bounds sent as naive-local ISO), `History.tsx` (filter bar + Timeline/Gallery tabs + infinite scroll; server-side filtering; dedupes appended pages by `id`), `Statistics.tsx` (range + day/week toggles feeding the Nivo charts; stale-while-revalidate), `Camera.tsx` (Test snapshot + `CropEditor`), `Hardware.tsx` (`SystemMonitor` + `NetworkMonitor`), `Settings.tsx` (grouped setting cards, saves only changed fields, restart badge + "Apply & restart" banner).
- Components: `DetectionCard` (thumbnail + both confidences + corrected mark; responsive full-width row on mobile / vertical plate on `sm`+), `Lightbox` (full-screen overlay: still + specimen-label caption, box-overlay toggle, Photo/Video toggle, Correct-ID pencil → `SpeciesPicker`, Field-guide + Advanced-stats panels — side-by-side on desktop, stacked segmented control below `lg`; synchronised image swap on prev/next; prev/next via edge arrows + ←/→ on desktop, horizontal image **swipe** + a top-centre "N / total" position chip on mobile — the floating arrows are `lg`-only, driven by the optional `position` prop the list-owning parents pass), `SpeciesPicker` (keyboard type-ahead over the vocabulary + add-a-new-species), plus `TodaySummary` (day-totals panel — sightings/species/most-common)/`ExpectedThisWeek`/`FeaturedDetection` (hero image + per-sighting stats)/`AdvancedStats`/`SystemMonitor`/`NetworkMonitor`/`CropEditor`/`Timeline`/`Gallery`/`FileDownloader`, and the Nivo charts under `components/charts/` with shared `charts/theme.ts` + `charts/format.ts` (CVD-validated 6-hue categorical palette — 6 is the safe ceiling, which is why the timeline requests top-6 + `Other`).
- Build: `npm run build` → `frontend/dist/` (served by FastAPI at `/`). Dev: `npm run dev` proxies `/api/*` to `http://localhost:8080`. `Dockerfile.api` is a multi-stage image (Node builds the frontend, Python runs the API).

### Model files (not in repo, must exist on the Pi)

Resolved by `detector/paths.py` (env-overridable via `ASSETS_DIR`/`MODEL_DIR`).

| Purpose | Path (relative to repo root) |
|---|---|
| Object detection (IMX500 firmware) | `/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk` |
| Species classifier | `assets/models/convnext_v2_tiny_int8.onnx` |
| Class-to-index mapping | `assets/models/convnext_v2_tiny.onnx_class_to_idx.json` |
| COCO labels | `assets/labels/coco_labels.txt` |
| Geomodel (spatio-temporal prior) | `assets/models/BirdNET+_Geomodel_V3.0.3_Global_12K_FP32.onnx` (absent → `refresh_geo_priors` warns, startup continues) |
| Geomodel labels | `assets/labels/BirdNET+_Geomodel_V3.0.3_Global_12K_Labels.txt` (tracked) |
| Geomodel↔classifier crosswalk | `assets/labels/geomodel_classifier_map.json` (tracked, curated) |
| Off-Pi emulator detector (dev-only) | `assets/models/yolo11n.onnx` (out-of-band; absent → emulator + tests skip) |

## Gotchas

### Bounding box format
All boxes are `(x, y, w, h)` in ISP output pixel coordinates after `imx500.convert_inference_coords`. `preprocess_roi` expands the box to a square with 20% padding before the classifier.

### DNN inference input vs. ScalerCrop — check this first if detection confidence is low
The IMX500 runs object detection **on-sensor**, before the ISP. `ScalerCrop` (the feeder crop) only crops the ISP `main` stream the **classifier** and saved stills read — it does **not** change what the **detector** sees. Left unrestricted, the DNN squishes the *whole* 4056×3040 field of view into its 640×640 input tensor, so a feeder bird is a tiny speck and `detection_confidence` collapses (~0.27 observed, vs ~0.7–0.8 when the bird fills the frame). This was the root cause of the "low YOLO confidence on the Pi" investigation — not orientation, colour, or int8 quantization (all ruled out via `notebooks/tracking_playground.ipynb` + the `tools/imx_emulator_probe.py` int8 emulator, which were fed the *cropped* clip, i.e. **not** the DNN's actual input). Fix: `build_camera` calls `imx500.set_inference_roi_abs(inference_roi_for_crop(region))` to restrict the DNN input to the crop region (default on; `config.restrict_inference_to_crop=False` reverts). The ROI is a **direct pass-through** of the crop rectangle in full-sensor pixels; if the DNN ever targets the diagonally-opposite region (detections vanish / confidence unchanged after enabling), the sensor ROI control uses the unflipped frame vs `ScalerCrop`'s flipped one — apply the 180° flip documented in `crop.inference_roi_for_crop`. Aspect note: a non-square crop is squished into the square input tensor (harmless for the default 900×900 feeder crop).

**Restricting the ROI also changes the network's *output* coordinates — check this if boxes are misplaced / classification drops after enabling the crop.** With an inference ROI set, the on-chip network returns boxes normalized `[0,1]` **relative to the ROI**, but `imx500.convert_inference_coords` treats them as fractions of the **full sensor** (it only *clamps* to the ROI via `bounded_to`, never translating/scaling by it). Un-remapped, every box is mis-scaled and clamped to the ROI edge — the bird is still detected but the drawn/persisted box is in the wrong place, and the ROI cropped for the classifier misses the bird (confidence collapses). So `parse_detections` remaps each box from ROI space back to full-sensor fractions (`(roi_origin + frac·roi_size)/sensor`) **before** `convert_inference_coords`. The ROI used for the remap is published by `build_camera`'s `apply_inference_roi` into `Camera.inference_roi_state` (the *exact* rectangle pushed to the hardware, a single source of truth), and `_run_capture_loop` passes it to `parse_detections` (`None` = full FOV = no-op). The off-Pi emulator's `FakeIMX500.convert_inference_coords` models this full-sensor→`ScalerCrop`→`main` mapping so `tests/detector/test_capture_loop.py` genuinely exercises the round-trip.

### Colour channel order (RGB vs BGR) — check this first if colours invert
The `main` ISP stream is `"format": "BGR888"`, **not** `"RGB888"`. picamera2's `888` names are byte-reversed vs the numpy array: `"BGR888"` yields an `[R,G,B]` array. Everything downstream assumes **RGB** (the ConvNeXt ImageNet normalisation, `PIL.Image.fromarray`, the `cv2.cvtColor(..., RGB2BGR)` writes). Requesting `"RGB888"` swaps red↔blue in every saved image and feeds the classifier mis-ordered channels (garbage predictions).

### CMA / buffer_count / numpy pin (detector container)
- `main.py` configures the camera with `buffer_count=6` (not the bare-Pi example's 12). Inside the container the CMA/dma-heap pool is host-global and shared with the firmware upload + the always-allocated 2028×1520 raw stream; 12 buffers exhausted CMA and crashed `picam2.start()` with `OSError: [Errno 12] Cannot allocate memory`. If it ENOMEMs after the `DEFAULT_LONG_SIDE` 640→1280 bump, drop `DEFAULT_LONG_SIDE` back toward 640 or raise the host CMA pool in `/boot/firmware/config.txt` — do **not** drop `buffer_count`.
- `requirements.detector.txt` pins `numpy==1.24.2` (the version the apt-built picamera2/opencv stack compiles against) and `onnxruntime==1.23.2`. Pip must not pull numpy 2.x into `/usr/local` and shadow the apt copy — that crashes the detector with `numpy.dtype size changed`. These are pip-installed into the system interpreter with `--break-system-packages`.

## Conventions

- All functions must have type hints and docstrings; functions do one thing.
- One package, `birdscanner`, absolute imports; layering `detector → ml → db` / `api → db` is one-way (`ml/` never imports `detector/` or `api/`). Dev-only code lives under `tools/` (offline) and `dev/` (emulator) and is never imported by the services.
- Classified bird images write to `$IMAGE_DIR/{species}/` (default `/home/stefan/Pictures/bird_detections`), with a `_thumb.jpg` 200×200 thumbnail alongside.
- `db/` tests use SQLAlchemy `StaticPool` to share an in-memory SQLite connection across threads (the `engine` fixture in `tests/conftest.py`). `api/` tests override FastAPI deps via `app.dependency_overrides` (no real DB/filesystem needed).

## Deployment

Docker Compose stack (`docker-compose.yml`) with two services: `detector` (ML pipeline) and `api` (FastAPI + React frontend).

```bash
cp .env.example .env       # required before first run (git-ignored)
docker compose up --build  # first time
docker compose up -d       # normal
docker compose logs -f detector
```
UI at `http://birdpi.local:8080` (mDNS) or `http://<pi-ip>:8080`.

**Key points:**
- The `data` Docker volume is the **single source of truth**: `detector` writes the images + mp4 clips (under `$IMAGE_DIR/{species}/`), the SQLite DB, `crop.json` (`CROP_CONFIG_PATH`), and `settings.json` (`SETTINGS_PATH`); the `api` mounts it read-only and proxies all write actions (settings/crop/delete/correct) to the detector's control server.
- The `detector` image (`Dockerfile.detector`) is based on `dtcooper/raspberrypi-os:bookworm` so the system `python3` can import the apt-installed `python3-picamera2`/`python3-libcamera` (not on PyPI). numpy/opencv/pillow come from apt; `numpy==1.24.2` + `onnxruntime==1.23.2` + `sqlmodel` are pip-installed with `--break-system-packages`. A plain `python:3.x` base crashes with `ModuleNotFoundError: No module named 'libcamera'`.
- `detector` needs `privileged: true` (IMX500 access) and mounts `/run/udev:ro`, `/dev/dma_heap`, `/sys/kernel/debug`, and `/usr/share/imx500-models:ro` (the host's `.rpk` firmware, since the image's apt package doesn't ship it — without it the detector retry-loops on a missing `.rpk`).
- Model files (`convnext_v2_tiny_int8.onnx`, the geomodel FP32 ONNX) must be in the build context on the Pi — baked into the image, not in the repo. Missing classifier → onnxruntime `NO_SUCHFILE` at startup; missing geomodel → logged warning, runs on without a location prior.
- The species-reference bank rides on the data volume at `/data/species_reference` (`SPECIES_REFERENCE_DIR`), built offline (`tools/build_species_reference.py`) and copied over / built with `--output-dir` straight against the deployed bank; **restart the api** afterwards (the manifest is memoized per-directory). Absent → every Lightbox reference lookup 404s.
- The `detector` `expose`s port 8000 (control server) on the internal network only; the `api` reaches it via `DETECTOR_URL` (default `http://detector:8000`), the detector binds `CAMERA_SERVER_PORT` (default 8000) — change them together.
- **Video encoding is software** (no Pi 5 hardware encoder) via the apt opencv FFmpeg backend, encoding H.264 (`avc1`); Debian's libavcodec has libx264 so `avc1` is available. An `mp4v` fallback file won't play in the browser (shows only the poster).
- If the camera is unavailable the detector does **not** crash-loop (`wait_for_camera()` retries every 30 s); the `api` is independent and the UI still loads (the detector initialises the DB schema on startup).

### Remote access (Tailscale)

The UI ships **without authentication**, and the `api` proxies destructive actions to the detector (delete/correct detections, change settings, **restart**, re-crop). So the UI must never be exposed raw to the public internet (no router port-forward). To reach it off-LAN — e.g. from a phone on cellular to show friends — put the Pi on a **Tailscale** mesh VPN; tailnet membership becomes the missing auth layer. **No repo/compose changes** — port 8080 is already published to the host, so it's reachable over the tailnet as-is; port 8000 stays internal-only.

- **Install on the Pi host** (not in Docker, so MagicDNS names the Pi and the host's 8080 is reachable): `curl -fsSL https://tailscale.com/install.sh | sh` then `sudo tailscale up` and authenticate (that account owns the tailnet).
- **Enable MagicDNS** (admin console → DNS) for a stable name like `birdpi.<tailnet>.ts.net`.
- **Add other people/devices**: install the app + sign in on your phone; invite your brother to the tailnet, or share just the Pi node (admin console → Machines → Share) if he has his own tailnet. Friends viewing via your phone need nothing.
- **HTTPS + clean hostname** (recommended for mobile): `sudo tailscale serve --bg 8080` → `https://birdpi.<tailnet>.ts.net`.
- **Do NOT enable `tailscale funnel`** without first adding a real auth gate — it publishes the endpoint publicly and removes the tailnet auth boundary in front of the unauthenticated, destructive-action UI.
- **Verify** it's true remote access, not LAN: load the URL from a phone on cellular (Wi-Fi off); confirm a non-tailnet device can't connect.
