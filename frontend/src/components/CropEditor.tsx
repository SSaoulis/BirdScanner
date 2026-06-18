import { useCallback, useEffect, useRef, useState } from "react";
import { api, type CropState, type NormalizedBox } from "../api";

/** Smallest allowed box edge as a fraction of the preview. */
const MIN_SIZE = 0.04;

/** The four draggable corner handles. */
const CORNERS = ["nw", "ne", "sw", "se"] as const;
type Corner = (typeof CORNERS)[number];

type DragState =
  | { mode: "move"; startX: number; startY: number; startBox: NormalizedBox }
  | { mode: "resize"; corner: Corner; startX: number; startY: number; startBox: NormalizedBox }
  | null;

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

/**
 * Interactive detection-region editor.
 *
 * Loads a full-sensor preview from the detector and overlays a draggable /
 * resizable box. The box is tracked in normalized [0, 1] coordinates over the
 * displayed (180°-flipped) preview — exactly the space the detector expects —
 * so applying it sends `{ nx, ny, nw, nh }` straight to `POST /api/camera/crop`.
 * The preview is rendered at the true sensor aspect ratio so the overlay maps
 * linearly to the sensor.
 */
export function CropEditor() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [box, setBox] = useState<NormalizedBox | null>(null);
  const [aspect, setAspect] = useState<number>(4056 / 3040);
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const objectUrlRef = useRef<string | null>(null);
  const dragRef = useRef<DragState>(null);

  const setPreviewBlob = useCallback((blob: Blob) => {
    const url = URL.createObjectURL(blob);
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = url;
    setPreviewUrl(url);
  }, []);

  /** Load the current crop region and a fresh full-sensor preview. */
  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setStatus(null);
    try {
      const crop: CropState = await api.camera.getCrop();
      setBox(crop.norm);
      if (crop.sensor_w > 0 && crop.sensor_h > 0) {
        setAspect(crop.sensor_w / crop.sensor_h);
      }
      const res = await fetch(`${api.camera.fullSnapshotUrl()}?t=${Date.now()}`);
      if (!res.ok) throw new Error(`Preview returned ${res.status} ${res.statusText}`);
      setPreviewBlob(await res.blob());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load crop editor");
    } finally {
      setLoading(false);
    }
  }, [setPreviewBlob]);

  useEffect(() => {
    void load();
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, [load]);

  const onPointerMove = useCallback((e: PointerEvent) => {
    const drag = dragRef.current;
    const container = containerRef.current;
    if (!drag || !container) return;
    const rect = container.getBoundingClientRect();
    const dx = (e.clientX - drag.startX) / rect.width;
    const dy = (e.clientY - drag.startY) / rect.height;
    const b = drag.startBox;

    if (drag.mode === "move") {
      setBox({
        nx: clamp(b.nx + dx, 0, 1 - b.nw),
        ny: clamp(b.ny + dy, 0, 1 - b.nh),
        nw: b.nw,
        nh: b.nh,
      });
      return;
    }

    let { nx, ny, nw, nh } = b;
    const right = b.nx + b.nw;
    const bottom = b.ny + b.nh;
    if (drag.corner.includes("e")) nw = clamp(b.nw + dx, MIN_SIZE, 1 - b.nx);
    if (drag.corner.includes("s")) nh = clamp(b.nh + dy, MIN_SIZE, 1 - b.ny);
    if (drag.corner.includes("w")) {
      nx = clamp(b.nx + dx, 0, right - MIN_SIZE);
      nw = right - nx;
    }
    if (drag.corner.includes("n")) {
      ny = clamp(b.ny + dy, 0, bottom - MIN_SIZE);
      nh = bottom - ny;
    }
    setBox({ nx, ny, nw, nh });
  }, []);

  const endDrag = useCallback(() => {
    dragRef.current = null;
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", endDrag);
  }, [onPointerMove]);

  const beginDrag = useCallback(
    (e: React.PointerEvent, drag: NonNullable<DragState>) => {
      e.preventDefault();
      e.stopPropagation();
      dragRef.current = drag;
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", endDrag);
    },
    [onPointerMove, endDrag],
  );

  useEffect(() => endDrag, [endDrag]);

  /** Persist the current box (or reset to default) and reload. */
  const apply = useCallback(
    async (body: NormalizedBox | { reset: true }) => {
      setApplying(true);
      setError(null);
      setStatus(null);
      try {
        const crop = await api.camera.setCrop(body);
        setBox(crop.norm);
        if (crop.sensor_w > 0 && crop.sensor_h > 0) {
          setAspect(crop.sensor_w / crop.sensor_h);
        }
        setStatus(
          `Detection region set to ${crop.w}×${crop.h}px at (${crop.x}, ${crop.y}).`,
        );
        // Pull a fresh preview so the live (cropped) feed change is reflected.
        const res = await fetch(`${api.camera.fullSnapshotUrl()}?t=${Date.now()}`);
        if (res.ok) setPreviewBlob(await res.blob());
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to apply crop");
      } finally {
        setApplying(false);
      }
    },
    [setPreviewBlob],
  );

  const busy = loading || applying;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={() => void load()}
          disabled={busy}
          className="rounded-lg border border-line bg-card px-3 py-2 text-sm font-semibold text-bark transition-colors hover:text-ink disabled:opacity-50"
        >
          {loading ? "Loading…" : "Reload preview"}
        </button>
        <button
          onClick={() => box && void apply(box)}
          disabled={busy || !box}
          className="rounded-lg bg-gold px-3 py-2 text-sm font-semibold text-card shadow-sm transition-colors hover:bg-gold-deep disabled:opacity-50"
        >
          {applying ? "Applying…" : "Apply region"}
        </button>
        <button
          onClick={() => void apply({ reset: true })}
          disabled={busy}
          className="rounded-lg border border-line bg-card px-3 py-2 text-sm font-semibold text-bark transition-colors hover:text-ink disabled:opacity-50"
        >
          Reset to feeder
        </button>
      </div>

      <p className="text-sm text-bark">
        Drag the box to move it, or pull a corner to resize. Whatever sits inside the
        box is the patch the detector watches for birds. Applying it briefly
        interrupts the live feed while the camera resets.
      </p>

      {error && (
        <div className="rounded-lg border border-rust/40 bg-rust/10 px-4 py-3 text-sm text-rust">
          {error}
        </div>
      )}
      {status && (
        <div className="rounded-lg border border-sage/50 bg-sage/15 px-4 py-3 text-sm text-sage-deep">
          {status}
        </div>
      )}

      <div className="rounded-2xl border border-line bg-card p-4 shadow-plate">
        <div
          ref={containerRef}
          className="relative mx-auto w-full max-w-2xl select-none overflow-hidden rounded-lg bg-ink"
          style={{ aspectRatio: `${aspect}` }}
        >
          {previewUrl && (
            <img
              src={previewUrl}
              alt="Full sensor preview"
              draggable={false}
              className="absolute inset-0 h-full w-full"
              style={{ objectFit: "fill" }}
            />
          )}

          {box && (
            <div
              onPointerDown={(e) =>
                beginDrag(e, {
                  mode: "move",
                  startX: e.clientX,
                  startY: e.clientY,
                  startBox: box,
                })
              }
              className="absolute cursor-move border-2 border-gold bg-gold/15"
              style={{
                left: `${box.nx * 100}%`,
                top: `${box.ny * 100}%`,
                width: `${box.nw * 100}%`,
                height: `${box.nh * 100}%`,
              }}
            >
              {CORNERS.map((corner) => (
                <span
                  key={corner}
                  onPointerDown={(e) =>
                    beginDrag(e, {
                      mode: "resize",
                      corner,
                      startX: e.clientX,
                      startY: e.clientY,
                      startBox: box,
                    })
                  }
                  className="absolute h-3 w-3 rounded-sm border border-card bg-gold"
                  style={{
                    left: corner.includes("w") ? "-6px" : undefined,
                    right: corner.includes("e") ? "-6px" : undefined,
                    top: corner.includes("n") ? "-6px" : undefined,
                    bottom: corner.includes("s") ? "-6px" : undefined,
                    cursor: `${corner}-resize`,
                  }}
                />
              ))}
            </div>
          )}

          {!previewUrl && (
            <div className="absolute inset-0 flex items-center justify-center">
              <p className="text-sm text-bark">
                {loading ? "Loading preview…" : "No preview available."}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
