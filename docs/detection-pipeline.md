# Detection Pipeline — End to End

This document traces a single frame from the camera all the way to a persisted,
classified bird detection. It is written for a machine-learning engineer who
wants to **add new functionality** to the pipeline (a new gating rule, a second
model, extra persisted metadata, a new side-effect on classification, etc.). It
names every function, the objects they produce, and where state is stored.

> **Scope.** The runtime path is `detector → ml → db`. `ml/` is
> platform-independent (no `picamera2`); the Pi-only camera glue lives in
> `detector/`. All boxes are `(x, y, w, h)` in ISP-output pixel coordinates
> unless stated otherwise.

---

## 1. High-level flow

```
main()                                         birdscanner/detector/main.py
 ├─ configure_logging(debug)                   detector/track_logging.py
 ├─ make_engine() / init_db(engine)            db/database.py         → SQLite schema
 ├─ wait_for_camera(model)  → IMX500           detector/camera.py
 ├─ prepare_intrinsics(imx500) → intrinsics    detector/camera.py
 ├─ setup_classifier(...)   → Classifier       ml/classification_pipeline.py
 ├─ build_gating(intrinsics) → Gating          detector/gating.py
 ├─ DetectionWriter(session_factory)           db/writer.py
 ├─ build_manager(classifier, gating, writer)  detector/gating.py
 │      → PipelineContext + ClassificationManager
 ├─ build_camera(imx500, intrinsics) → Camera  detector/camera.py
 ├─ _start_control_server(camera, engine)      detector/main.py (+ camera_server.py)
 └─ _run_capture_loop(camera, manager, gating) detector/main.py
        every frame:
          detection_callback(request)          [picam2 pre_callback]
            ├─ tracker.update_frame(last_results)
            └─ process_detections(request, "main", last_results, manager, labels)
          capture loop body:
            └─ parse_detections(metadata, imx500, intrinsics, threshold, picam2)
                  → list[Detection]  (stored in state["last_results"])
```

Two things run per frame, on **different threads**:

1. **The capture loop** (`_run_capture_loop`, main thread) calls
   `capture_metadata()` then `parse_detections(...)`, storing the result in
   `state["last_results"]`.
2. **The picamera2 `pre_callback`** (`detection_callback`, camera thread) runs
   for each request: it advances the tracker with the *previous* frame's results
   and calls `process_detections(...)`, which draws boxes and **queues** bird
   detections onto the `ClassificationManager`.

The actual classification then happens on **a third thread** (the
`ClassificationManager` worker) when `use_multithreading=True` (the default).

---

## 2. Startup objects (built once in `main()`)

| Object | Built by | Purpose |
|---|---|---|
| `engine` | `make_engine()` | SQLite engine; `init_db()` creates/migrates the `detections` table. Detector owns all writes. |
| `imx500` | `wait_for_camera(model)` | IMX500 device handle; retry-with-backoff so a missing camera never crash-loops. |
| `intrinsics` | `prepare_intrinsics(imx500)` | Network intrinsics (labels, fps, bbox order/normalization). |
| `Classifier` | `setup_classifier(model_path, class_to_idx_path)` | ConvNeXt V2 Tiny ONNX wrapper + preprocessing (384×384, ImageNet norm). |
| `Gating` | `build_gating(intrinsics)` | Bundle of `tracker`, `best_frame_selector`, `video_recorder`. |
| `DetectionWriter` | `DetectionWriter(make_session_factory(engine))` | Fire-and-forget background DB writer. |
| `PipelineContext` + `ClassificationManager` | `build_manager(classifier, gating, writer)` | The per-detection dependency bundle + dispatcher. |
| `Camera` | `build_camera(imx500, intrinsics)` | picam2 + imx500 + intrinsics + `CropController` bundle. |

### `Gating` (`detector/gating.py`)

```python
@dataclass
class Gating:
    tracker: StableDetectionTracker
    best_frame_selector: BestFrameSelector
    video_recorder: Optional[VideoRecorder]
```

`build_gating` computes `min_stable_frames(fps)` from
`config.object_duration_threshold × fps`, wires the tracker's
`on_track_became_stable`/`on_track_deleted` callbacks (logging + freeing the
best frame), and only creates a `VideoRecorder` when `config.video.save` is set.

> **Legacy mode.** If `object_duration_threshold <= 0`, gating is per-frame and
> the whole `Gating` bundle's components are effectively disabled. The default
> config uses stable-track gating (`0.2 s`), which is the path documented here.

### `PipelineContext` (`ml/classification_pipeline.py`)

The single object every per-detection function receives. **This is the seam to
extend** — add a dependency here and it is available everywhere downstream:

```python
@dataclass
class PipelineContext:
    classifier: Classifier
    tracker: StableDetectionTracker          # defaults to module global
    classify_fn: ClassifyFn                   # defaults to run_bird_classification
    detection_writer: Optional[DetectionWriter] = None
    best_frame_selector: Optional[BestFrameSelector] = None
    record_fn: Optional[Callable[[str], bool]] = None       # VideoRecorder.trigger
    video_frame_fn: Optional[Callable[[np.ndarray], None]] = None  # VideoRecorder.add_frame
```

`build_manager` injects `record_fn`/`video_frame_fn` from the `VideoRecorder`,
keeping picamera2/video code out of `ml/`.

---

## 3. Per-frame path

### 3.1 `parse_detections(...)` → `list[Detection]`

`birdscanner/ml/object_detection.py`

- Reads the inference tensor via `imx500.get_outputs(metadata, add_batch=True)`.
  Returns the module-global `last_detections` fallback when a frame has no
  output (so the live loop always has something to draw).
- `_decode_boxes` normalizes and re-orders the raw box tensor per `intrinsics`.
- For each box with `score > threshold`, builds a **`Detection`** and calls
  `set_box(imx500.convert_inference_coords(...))` to convert to ISP pixels.

```python
class Detection:
    category: int          # class index into labels
    conf: float            # YOLO object-detection confidence [0,1]
    box: tuple | None      # (x, y, w, h) in ISP output pixels, set by set_box()
```

`get_labels(intrinsics)` returns the filtered label list (drops `""` and `"-"`).

### 3.2 Tracking — `StableDetectionTracker.update_frame(...)`

`birdscanner/ml/tracking.py`

Called each frame from `detection_callback` with the previous frame's detection
list. IoU-based, pure-function core (`match_detection_to_track`,
`update_tracks_for_frame`, `should_classify_track`):

```python
@dataclass
class StableTrack:
    track_id: int
    box: tuple
    stable_frames: int = 1
    classified: bool = False
    frames_since_seen: int = 0
    species: Optional[str] = None
```

- A detection matches an existing track when IoU > `iou_threshold` (0.6).
  Matched tracks increment `stable_frames`; unmatched detections spawn a new
  track. Tracks unseen for `max_missing_frames` are deleted (firing
  `on_track_deleted`).
- `update_frame` fires `on_track_became_stable(track)` **once**, the frame a
  track first reaches `min_stable_frames`.
- `track_for_detection_id(detection_id)` maps a frame-local detection index back
  to its `StableTrack` (used by the classification path). `mark_classified`
  sets `classified=True` so each track is classified **at most once**.

Gating predicate:
`should_run_bird_classification_for_detection(detection_id, tracker=...)` →
`True` only when the detection's track exists, is stable, and is not yet
classified.

### 3.3 `process_detections(...)` — draw + queue

`birdscanner/ml/classification_pipeline.py` (picam2 `pre_callback` entry point)

For each `Detection` in `last_results`:

1. `label_for_category(labels, detection.category)` — bounds-checked; a spurious
   out-of-range category is logged and **skipped** (never crashes the callback).
2. `preprocess_roi(full_img, detection.box)` → `(roi, coords)`; `draw_boxes(...)`
   annotates a copy written back to `m.array` (**live preview only** — the saved
   still is clean).
3. Every clean frame is fed to `context.video_frame_fn` (the pre-roll ring
   buffer) — cheap, no encoding while idle.
4. If the label is `"bird"`:
   - `_observe_best_frame(...)` → `best_frame_selector.observe(track_id, frame,
     box, detection.conf)` retains the highest-YOLO-confidence frame per track.
   - `manager.process((full_img, detection_id, detection, labels,
     classifier_class))` — **queues** the item.

---

## 4. Classification & persistence

### 4.1 Dispatch — `ClassificationManager`

`birdscanner/ml/classification_pipeline.py`

- `process(item)` — sync mode runs immediately; async mode `put_nowait`s onto a
  `Queue` and **drops** the item if full (so the camera callback never blocks).
- `_dispatch(item)` — **wraps processing in `try/except` and logs+swallows any
  exception**, so one bad detection can never kill the worker thread. (An
  unhandled exception here previously killed the worker permanently, after which
  nothing was ever classified again.)
- `_worker_loop` — drains the queue; `task_done()` in a `finally` so it never
  wedges. `stop()` enqueues a sentinel and joins.

### 4.2 The core function — `process_single_detection_with_stable_tracks(item, context)`

Item tuple: `(image, detection_id, detection, labels, classifier_class)`.

```
is_bird = classifier_class.lower() == "bird"
still   = Still(image, detection.box)          # NamedTuple(frame, box)

if is_bird and should_run_bird_classification_for_detection(detection_id, tracker):
    track  = tracker.track_for_detection_id(detection_id)
    still  = _best_still(context, track, still)          # swap in best frame
    result = _classify_track(context, still, detection_id, track)  # Classification | None

if is_bird and result and result.confidence > _SAVE_CONFIDENCE_THRESHOLD (0.4):
    _persist_detection(context, still, detection, track, result)
```

Helper functions (each does one thing — the seams for new behavior):

| Function | Produces | Notes |
|---|---|---|
| `_best_still(context, track, still)` | `Still` | `best_frame_selector.take(track_id)`, else the trigger frame. |
| `_classify_track(context, still, id, track)` | `Classification` or `None` | `preprocess_roi` → **skips empty ROI** (`roi.size == 0`) without marking the track; else `classify_fn(classifier, roi)` and `tracker.mark_classified(track_id, species)`. |
| `_save_still_and_thumbnail(species_dir, stem, still)` | files on disk | Writes clean `{stem}.png` (RGB→BGR via cv2) + `{stem}_thumb.jpg` (200×200). |
| `_start_clip(context, species_dir, stem, species)` | `(video_path, no_video_reason)` | `record_fn` → `video_path` when recording began; else `("disabled"`/`"recorder_busy")`. |
| `_persist_detection(...)` | log lines + DB row | Builds `DetectionRecord`, calls `detection_writer.write(record)`. |

`Still` and `Classification` are the small value objects that keep signatures
short:

```python
class Still(NamedTuple):          # a candidate frame + its box
    frame: np.ndarray
    box: tuple

class Classification(NamedTuple): # a species prediction
    species: str
    confidence: float
```

### 4.3 Best-frame selection — `BestFrameSelector`

`birdscanner/ml/best_frame.py` (pure numpy, thread-safe)

- `observe(track_id, frame, box, score)` — camera thread; keeps the single
  highest-`score` (`detection.conf`) `BestCandidate` per track.
- `take(track_id)` — classifier thread; pops the retained candidate.
- `discard(track_id)` — freed when the track ends via `on_track_deleted`.

So the saved still/thumbnail/box come from the **clearest frame across the
track**, not the arbitrary frame that triggered classification.

### 4.4 The classifier — `Classifier.classify(roi)`

`birdscanner/ml/classification.py`

`setup_classifier(model_path, class_to_idx_path)` builds a `Classifier` around
an `ONNXClassifier` with `build_preprocessing` (resize 384×384 → center crop →
ImageNet normalize → NCHW). `classify(image)` returns `(species_str,
confidence_float)`.

> **Colour order matters:** the `main` stream is `BGR888`, which yields an
> `[R,G,B]` array. Everything downstream assumes RGB. Feeding BGR to the
> classifier produces garbage predictions — see CLAUDE.md "Colour channel
> order".

### 4.5 Persistence — `DetectionRecord` + `DetectionWriter`

`_persist_detection` builds a `DetectionRecord` (`birdscanner/db/models.py`,
table `detections`) and hands it to `DetectionWriter.write(record)`.

**Objects stored per detection:**

- **On disk** under `$IMAGE_DIR/{species}/`:
  - `{stem}.png` — clean full still (no box drawn).
  - `{stem}_thumb.jpg` — 200×200 thumbnail.
  - `{stem}.mp4` — short clip (pre+post-roll), finishes encoding a few seconds
    *after* the row is written (so `/video` briefly 404s).
- **In SQLite** (`DetectionRecord`), the columns most relevant to extension:

  | Column | Source |
  |---|---|
  | `timestamp` | `datetime.now()` (naive local) — the `stem` derives from it. |
  | `species` / `confidence` | `Classification` (ConvNeXt). |
  | `detection_confidence` | `detection.conf` (YOLO). |
  | `image_path` / `thumbnail_path` / `video_path` | relative to `IMAGE_DIR`. |
  | `no_video_reason` | `"recorder_busy"` / `"disabled"` / `None`. |
  | `track_id` / `stable_frames` | from the `StableTrack`. |
  | `box_x/box_y/box_w/box_h` | `normalized_box(still.box, still.frame.shape)` — `[0,1]` fractions. |
  | `duration_sec` / `uploaded_at` | currently unused (nullable). |

`DetectionWriter` (`db/writer.py`) enqueues onto a bounded `Queue` (maxsize 64)
and commits on a daemon thread; a full queue **drops** the record (never blocks
the caller). `stop()` flushes via a sentinel.

> **Adding a DB column:** add the field to `DetectionRecord`, then register it in
> `_DETECTIONS_ADDED_COLUMNS` in `db/database.py` so `_migrate_detections_columns`
> backfills existing DBs (`create_all` never alters an existing table).

---

## 5. Where to plug in new functionality

| You want to… | Do this |
|---|---|
| Add a per-detection dependency (2nd model, uploader, metrics sink) | Add a field to **`PipelineContext`**; inject it in `build_manager`; use it in `process_single_detection_with_stable_tracks` / its helpers. |
| Change *when* a track classifies | Adjust `StableDetectionTracker` params (`iou_threshold`, `min_stable_frames`) in `build_gating`, or the `should_classify_track` logic. |
| Swap/second the classifier | Override `context.classify_fn` (signature `(Classifier, np.ndarray) -> (str, float)`), or add a new callable to the context. |
| Persist new metadata | Add a column to `DetectionRecord` + `_DETECTIONS_ADDED_COLUMNS`; populate it in `_persist_detection`'s `DetectionRecord(...)`. |
| React to a new best-frame policy | Change the `score` passed to `best_frame_selector.observe` in `_observe_best_frame` (currently `detection.conf`). |
| Add a side effect on classification (e.g. notify) | Add a helper called from `_persist_detection`, gated on `result.confidence`. |
| Change the save threshold | `_SAVE_CONFIDENCE_THRESHOLD` in `classification_pipeline.py`. |

**Concurrency contract to respect:** `process_detections`/`observe` run on the
camera thread; classification/`take`/`write` run on the worker thread. Anything
shared across the two must be thread-safe (the selector and writer already are).
Keep heavy work off the camera callback — queue it via `manager.process`.

---

## 6. Key files

| Concern | File |
|---|---|
| Entry point + capture loop | `birdscanner/detector/main.py` |
| Camera bring-up | `birdscanner/detector/camera.py` |
| Gating wiring | `birdscanner/detector/gating.py` |
| Detection parsing / `Detection` | `birdscanner/ml/object_detection.py` |
| Tracking | `birdscanner/ml/tracking.py` |
| Classification orchestration + persistence | `birdscanner/ml/classification_pipeline.py` |
| Best-frame selection | `birdscanner/ml/best_frame.py` |
| ONNX classifier | `birdscanner/ml/classification.py` |
| Geometry / drawing helpers | `birdscanner/ml/detection_utils.py` |
| DB model | `birdscanner/db/models.py` |
| DB writer | `birdscanner/db/writer.py` |
| Video recorder | `birdscanner/detector/video_recorder.py` |
</content>
