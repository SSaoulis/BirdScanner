import { useEffect, useRef, useState } from "react";
import {
  api,
  timeAgo,
  ApiError,
  type Detection,
  type SpeciesReference,
} from "../api";

interface LightboxProps {
  /** The detection currently displayed in the panel. */
  detection: Detection;
  /** Called when the panel should close. */
  onClose: () => void;
  /** Called to navigate to the previous detection. Pass null when at the start. */
  onPrev: (() => void) | null;
  /** Called to navigate to the next detection. Pass null when at the end. */
  onNext: (() => void) | null;
  /** Called with the detection id after it has been successfully deleted. */
  onDelete: (id: number) => void;
}

/**
 * Human-readable explanation for why a sighting has no video clip, shown as the
 * tooltip on the disabled Video toggle. Mirrors the `no_video_reason` values the
 * detector persists (see `birdscanner/ml/classification_pipeline.py`).
 */
function noVideoReasonText(reason: string | null): string {
  switch (reason) {
    case "recorder_busy":
      return "No clip — the recorder was busy saving another sighting's video. Only one clip records at a time to spare the Pi's CPU, so this sighting overlapped another recording.";
    case "disabled":
      return "No clip — video recording is turned off.";
    default:
      return "No clip available for this sighting.";
  }
}

/** Status of the reference fetch for the current species. */
type ReferenceState =
  | { kind: "loading" }
  | { kind: "ready"; reference: SpeciesReference }
  | { kind: "none" }
  | { kind: "error"; message: string };

/**
 * Full-screen lightbox opened when a detection thumbnail is clicked.
 *
 * Shows the captured detection image. A vertical "Reference" tab on the right
 * edge of the image toggles a species-reference panel that opens to the *exact*
 * rendered size of the image (measured live) and never grows beyond it — its
 * content scrolls internally instead, so the detection image is never offset.
 * Reference data is fetched from `/api/species/{name}/reference`. Supports
 * keyboard navigation (Esc, ArrowLeft, ArrowRight), prev/next arrow buttons,
 * backdrop-click to close, and locks body scroll while open. Navigating
 * prev/next changes the detection — and therefore its species — so the
 * reference is refetched whenever the displayed species changes.
 */
export function Lightbox({ detection, onClose, onPrev, onNext, onDelete }: LightboxProps) {
  const { id, species, confidence, detection_confidence, timestamp } = detection;
  const fullUrl = api.images.fullUrl(id);
  const thumbUrl = api.images.thumbnailUrl(id);
  const videoUrl = api.images.videoUrl(id);
  // A clip exists once video_path is set; legacy/disabled/still-encoding rows are
  // null. The Photo/Video toggle is always shown, but the Video button is
  // disabled (with a reason tooltip) when no clip is available.
  const hasVideo = detection.video_path !== null;
  // Which media the main pane shows. Reset to the still whenever the detection
  // changes so navigating to a clip-less sighting never lands on a blank player.
  const [mode, setMode] = useState<"photo" | "video">("photo");
  useEffect(() => {
    setMode("photo");
  }, [id]);
  const confidencePct = (confidence * 100).toFixed(1);
  const detectionPct =
    detection_confidence != null ? (detection_confidence * 100).toFixed(1) : null;

  // A persisted detection box (normalized [0, 1]) lets us overlay the bounding
  // box on the otherwise-clean saved image. Legacy rows predate this and have
  // null coordinates, so the toggle/overlay are hidden for them.
  const hasBox =
    detection.box_x !== null &&
    detection.box_y !== null &&
    detection.box_w !== null &&
    detection.box_h !== null;
  // Visible by default — toggled off to inspect the clean image.
  const [showBox, setShowBox] = useState(true);

  const [refState, setRefState] = useState<ReferenceState>({ kind: "loading" });
  // Index of the reference image shown prominently (for the multi-image case).
  const [activeImageIndex, setActiveImageIndex] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // Whether the reference panel is open. Closed by default — opened via the tab.
  const [showReference, setShowReference] = useState(false);
  // Live rendered size of the detection image; the reference panel is locked
  // to these exact pixel dimensions so it always matches the image.
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);

  // Track the rendered image size so the reference panel can match it exactly.
  // A ResizeObserver catches both the initial load (0 → natural size) and any
  // viewport resize that reflows the vw-capped image.
  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;
    const update = () => {
      if (el.clientWidth > 0 && el.clientHeight > 0) {
        setImgSize({ w: el.clientWidth, h: el.clientHeight });
      }
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [fullUrl]);

  /** Confirm, delete the detection via the API, then notify the parent. */
  async function handleDelete() {
    if (deleting) return;
    if (!window.confirm(`Permanently delete this ${species} detection and its image?`)) {
      return;
    }
    setDeleting(true);
    setDeleteError(null);
    try {
      await api.detections.delete(id);
      onDelete(id);
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : "Delete failed");
      setDeleting(false);
    }
  }

  // Keyboard navigation
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && onPrev) onPrev();
      if (e.key === "ArrowRight" && onNext) onNext();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose, onPrev, onNext]);

  // Prevent body scroll while open
  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, []);

  // Fetch the species reference whenever the displayed species changes.
  useEffect(() => {
    let cancelled = false;
    setRefState({ kind: "loading" });
    setActiveImageIndex(0);

    api.species
      .reference(species)
      .then((reference) => {
        if (cancelled) return;
        setRefState({ kind: "ready", reference });
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 404) {
          setRefState({ kind: "none" });
        } else {
          setRefState({
            kind: "error",
            message: e instanceof Error ? e.message : "Failed to load reference",
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [species]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink/95 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`A closer look at ${species}`}
    >
      {/* Prev arrow */}
      {onPrev && (
        <button
          className="absolute left-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-card/90 hover:bg-card text-ink text-2xl shadow-plate transition-colors z-10"
          onClick={(e) => { e.stopPropagation(); onPrev(); }}
          aria-label="Previous sighting"
        >
          &#8592;
        </button>
      )}

      {/* Image + reference row — stops click propagation so interacting inside doesn't close */}
      <div
        className="relative flex items-start"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Captured detection image (with the Reference tab on its edge) ── */}
        <div className="flex flex-col gap-3">
          <div className="relative">
            {mode === "video" && hasVideo ? (
              <video
                src={videoUrl}
                poster={thumbUrl}
                controls
                autoPlay
                loop
                muted
                className="block max-h-[80vh] max-w-[44vw] rounded-lg bg-ink shadow-plate-lift"
              />
            ) : (
              <img
                ref={imgRef}
                src={fullUrl}
                alt={`Captured ${species}`}
                className="block max-h-[80vh] max-w-[44vw] rounded-lg bg-ink shadow-plate-lift"
              />
            )}

            {/* Detection box overlay — positioned in normalized [0,1] space over
                the rendered image, so it scales with whatever size the image is
                capped to. Only meaningful on the still, so it is hidden in video
                mode, when toggled off, or for legacy boxless rows. */}
            {mode === "photo" && hasBox && showBox && (
              <div
                className="pointer-events-none absolute rounded-sm border-2 border-gold shadow-[0_0_0_1px_rgba(0,0,0,0.45)]"
                style={{
                  left: `${detection.box_x! * 100}%`,
                  top: `${detection.box_y! * 100}%`,
                  width: `${detection.box_w! * 100}%`,
                  height: `${detection.box_h! * 100}%`,
                }}
                aria-hidden="true"
              />
            )}

            {/* Close button */}
            <button
              className="absolute top-2 right-2 z-10 p-1.5 rounded-full bg-card/90 hover:bg-card text-ink text-lg leading-none shadow-plate"
              onClick={onClose}
              aria-label="Close"
            >
              ✕
            </button>

            {/* Vertical Reference tab on the right edge of the image */}
            <button
              className={`absolute top-1/2 left-full -translate-y-1/2 px-1.5 py-3 rounded-r-lg text-xs font-semibold uppercase tracking-wide [writing-mode:vertical-rl] transition-colors ${
                showReference
                  ? "bg-gold text-card"
                  : "bg-card/90 text-ink hover:bg-card"
              }`}
              onClick={() => setShowReference((v) => !v)}
              aria-pressed={showReference}
              aria-label={showReference ? "Hide field guide" : "Show field guide"}
            >
              Field guide
            </button>
          </div>

          {/* Caption bar */}
          <div className="flex flex-wrap items-center gap-3 rounded-xl border border-line bg-card/95 px-3 py-2 text-sm">
            <span className="font-display text-base font-medium text-ink">{species}</span>
            <span className="tnum font-medium text-gold-deep" title="Species-classification confidence">
              {confidencePct}% match
            </span>
            {detectionPct !== null && (
              <span className="tnum text-bark" title="Object-detection confidence (YOLO)">
                {detectionPct}% spotted
              </span>
            )}
            <span className="text-bark">{timeAgo(timestamp)}</span>
            <div
              className="flex overflow-hidden rounded-md border border-line"
              role="group"
              aria-label="Choose media"
            >
              {(["photo", "video"] as const).map((m) => {
                // The Video button stays visible even without a clip, but is
                // disabled and explains why on hover. aria-disabled (not the
                // native `disabled` attribute) keeps the title tooltip firing —
                // browsers suppress hover events on natively-disabled buttons.
                const unavailable = m === "video" && !hasVideo;
                return (
                  <button
                    key={m}
                    className={`px-3 py-1.5 text-xs font-medium capitalize transition-colors ${
                      mode === m
                        ? "bg-gold text-card"
                        : "bg-paper text-ink hover:bg-card"
                    } ${unavailable ? "cursor-not-allowed opacity-40 hover:bg-paper" : ""}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (!unavailable) setMode(m);
                    }}
                    aria-pressed={mode === m}
                    aria-disabled={unavailable}
                    title={
                      unavailable
                        ? noVideoReasonText(detection.no_video_reason)
                        : undefined
                    }
                  >
                    {m}
                  </button>
                );
              })}
            </div>
            {mode === "photo" && hasBox && (
              <button
                className={`ml-auto rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                  showBox
                    ? "border-gold bg-gold text-card hover:brightness-110"
                    : "border-line bg-paper text-ink hover:bg-card"
                }`}
                onClick={(e) => { e.stopPropagation(); setShowBox((v) => !v); }}
                aria-pressed={showBox}
              >
                {showBox ? "Box on" : "Box off"}
              </button>
            )}
            <a
              href={mode === "video" && hasVideo ? videoUrl : fullUrl}
              download
              className={`rounded-md border border-line bg-paper px-3 py-1.5 text-xs font-medium text-ink transition-colors hover:bg-card${
                mode === "photo" && hasBox ? "" : " ml-auto"
              }`}
              onClick={(e) => e.stopPropagation()}
            >
              Download
            </a>
            <button
              className="rounded-md bg-rust px-3 py-1.5 text-xs font-medium text-card transition-colors hover:brightness-110 disabled:opacity-50"
              onClick={(e) => { e.stopPropagation(); handleDelete(); }}
              disabled={deleting}
            >
              {deleting ? "Deleting…" : "Delete"}
            </button>
            {deleteError && (
              <span className="w-full text-xs text-rust">{deleteError}</span>
            )}
          </div>
        </div>

        {/* ── Reference panel — locked to the image's exact rendered size ──
            Stays mounted (once the image size is known) so it can unfold and
            fold away smoothly: the outer wrapper animates its width, left
            margin (the gap to the image) and opacity, clipping the fixed-size
            inner card so its content never reflows mid-animation. The image is
            centred in the row, so it glides aside as the panel grows. The
            global prefers-reduced-motion guard zeroes these durations. */}
        {imgSize && (
          <div
            className="shrink-0 overflow-hidden transition-[width,margin-left,opacity] duration-300 ease-out motion-reduce:transition-none"
            style={{
              width: showReference ? imgSize.w : 0,
              marginLeft: showReference ? "2.5rem" : 0,
              opacity: showReference ? 1 : 0,
              pointerEvents: showReference ? "auto" : "none",
            }}
            aria-hidden={!showReference}
          >
            <div
              className="overflow-y-auto rounded-lg border border-line bg-card shadow-plate-lift p-4"
              style={{ width: imgSize.w, height: imgSize.h }}
            >
              <h3 className="eyebrow mb-3">Field guide</h3>
              <ReferencePane
                state={refState}
                activeImageIndex={activeImageIndex}
                onSelectImage={setActiveImageIndex}
              />
            </div>
          </div>
        )}
      </div>

      {/* Next arrow */}
      {onNext && (
        <button
          className="absolute right-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-card/90 hover:bg-card text-ink text-2xl shadow-plate transition-colors z-10"
          onClick={(e) => { e.stopPropagation(); onNext(); }}
          aria-label="Next sighting"
        >
          &#8594;
        </button>
      )}
    </div>
  );
}

interface ReferencePaneProps {
  /** Current fetch state for the species reference. */
  state: ReferenceState;
  /** Index of the reference image shown prominently. */
  activeImageIndex: number;
  /** Called when a thumbnail is selected. */
  onSelectImage: (index: number) => void;
}

/**
 * Renders the right-hand pane content for a species reference, handling the
 * loading / no-reference / error / ready states.
 */
function ReferencePane({ state, activeImageIndex, onSelectImage }: ReferencePaneProps) {
  if (state.kind === "loading") {
    return (
      <div className="flex flex-col items-center justify-center min-h-[200px] gap-3 text-bark">
        <div
          className="w-8 h-8 border-2 border-line border-t-gold rounded-full animate-spin"
          aria-hidden="true"
        />
        <p className="text-sm animate-pulse">Opening the guide…</p>
      </div>
    );
  }

  if (state.kind === "none") {
    return (
      <div className="flex items-center justify-center min-h-[200px] text-center px-4">
        <p className="text-sm text-bark">
          No guide entry for this species yet.
        </p>
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="flex items-center justify-center min-h-[200px] text-center px-4">
        <p className="text-sm text-rust">
          Couldn’t open the guide: {state.message}
        </p>
      </div>
    );
  }

  const { reference } = state;
  const safeIndex = Math.min(activeImageIndex, Math.max(reference.images.length - 1, 0));
  const activeImage = reference.images[safeIndex];

  return (
    <div className="flex flex-col gap-3">
      {/* Primary reference image */}
      {activeImage ? (
        <div className="flex flex-col gap-1">
          <img
            src={activeImage.url}
            alt={`Reference photo of ${reference.common_name}`}
            className="w-full object-contain rounded-lg bg-paper"
          />
          <p className="text-[11px] text-bark leading-snug">
            {activeImage.attribution}
            {activeImage.license ? ` · ${activeImage.license}` : ""}
          </p>
        </div>
      ) : (
        <div className="flex items-center justify-center h-32 rounded-lg bg-paper text-sm text-bark">
          No guide photo
        </div>
      )}

      {/* Thumbnail strip for additional images */}
      {reference.images.length > 1 && (
        <div className="flex flex-wrap gap-2" role="group" aria-label="Reference images">
          {reference.images.map((img, i) => (
            <button
              key={img.url}
              onClick={() => onSelectImage(i)}
              className={`w-14 h-14 rounded-md overflow-hidden border-2 transition-colors ${
                i === safeIndex
                  ? "border-gold"
                  : "border-transparent hover:border-line"
              }`}
              aria-label={`Show reference image ${i + 1}`}
              aria-pressed={i === safeIndex}
            >
              <img
                src={img.url}
                alt={`Reference thumbnail ${i + 1} of ${reference.common_name}`}
                className="w-full h-full object-cover"
              />
            </button>
          ))}
        </div>
      )}

      {/* Species info */}
      <div className="flex flex-col gap-2">
        <div>
          <p className="font-display text-lg font-medium text-ink">{reference.common_name}</p>
          {reference.scientific_name && (
            <p className="font-display text-sm italic text-bark">{reference.scientific_name}</p>
          )}
        </div>

        {reference.summary && (
          <p className="text-sm text-ink/85 leading-relaxed">{reference.summary}</p>
        )}

        {reference.behaviour && (
          <div>
            <p className="eyebrow mb-1">Behaviour</p>
            <p className="text-sm text-ink/85 leading-relaxed">{reference.behaviour}</p>
          </div>
        )}

        {reference.wikipedia_url && (
          <a
            href={reference.wikipedia_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-gold-deep hover:text-gold underline w-fit"
          >
            Read on Wikipedia →
          </a>
        )}
      </div>
    </div>
  );
}
