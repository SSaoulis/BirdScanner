import { useEffect, useRef, useState } from "react";
import {
  api,
  timeAgo,
  ApiError,
  type Detection,
  type SpeciesReference,
} from "../api";
import { SpeciesPicker } from "./SpeciesPicker";
import { AdvancedStatsPane } from "./AdvancedStats";
import { useIsDesktop } from "../hooks/useMediaQuery";

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
  /** Called with the updated detection after its species is corrected. */
  onUpdate: (updated: Detection) => void;
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
export function Lightbox({
  detection,
  onClose,
  onPrev,
  onNext,
  onDelete,
  onUpdate,
}: LightboxProps) {
  // The detection whose plate is *currently on screen*. Decoupled from the
  // incoming `detection` prop so that on prev/next we hold the current plate —
  // image, box overlay and caption, all mutually consistent — until the next
  // full-res image has decoded, then swap them together. Otherwise the box and
  // caption jump to the next sighting a beat before its (network-fetched) image
  // catches up, which reads as clunky. Everything below is derived from `shown`.
  const [shown, setShown] = useState<Detection>(detection);

  useEffect(() => {
    // Same record (e.g. an in-place species correction): the image bytes are
    // unchanged, so apply the update immediately — there is nothing to wait for.
    if (detection.id === shown.id) {
      setShown(detection);
      return;
    }
    // A different sighting: preload its full image off-screen and only swap the
    // visible plate once the bytes have decoded, so image + box + caption all
    // appear together. Falls through on error so a broken image still swaps
    // (showing its own broken state) instead of trapping the loader.
    let cancelled = false;
    const preload = new Image();
    const settle = () => {
      if (cancelled) return;
      setShown(detection);
    };
    preload.onload = settle;
    preload.onerror = settle;
    preload.src = api.images.fullUrl(detection.id);
    return () => {
      cancelled = true;
    };
  }, [detection, shown.id]);

  const { id, species, confidence, detection_confidence, timestamp } = shown;
  const corrected = shown.corrected === true;
  const originalSpecies = shown.original_species;
  const fullUrl = api.images.fullUrl(id);
  const thumbUrl = api.images.thumbnailUrl(id);
  const videoUrl = api.images.videoUrl(id);
  // A clip exists once video_path is set; legacy/disabled/still-encoding rows are
  // null. The Photo/Video toggle is always shown, but the Video button is
  // disabled (with a reason tooltip) when no clip is available.
  const hasVideo = shown.video_path !== null;
  // Which media the main pane shows. Reset to the still whenever the detection
  // changes so navigating to a clip-less sighting never lands on a blank player.
  const [mode, setMode] = useState<"photo" | "video">("photo");
  // Whether the species-correction picker is open, plus its in-flight/error state.
  const [correcting, setCorrecting] = useState(false);
  const [correctBusy, setCorrectBusy] = useState(false);
  const [correctError, setCorrectError] = useState<string | null>(null);
  useEffect(() => {
    setMode("photo");
    // Navigating to another sighting closes the picker so it never lingers over
    // the wrong record.
    setCorrecting(false);
    setCorrectBusy(false);
    setCorrectError(null);
  }, [id]);
  const confidencePct = (confidence * 100).toFixed(1);
  const detectionPct =
    detection_confidence != null ? (detection_confidence * 100).toFixed(1) : null;

  // A persisted detection box (normalized [0, 1]) lets us overlay the bounding
  // box on the otherwise-clean saved image. Legacy rows predate this and have
  // null coordinates, so the toggle/overlay are hidden for them.
  const hasBox =
    shown.box_x !== null &&
    shown.box_y !== null &&
    shown.box_w !== null &&
    shown.box_h !== null;
  // Visible by default — toggled off to inspect the clean image.
  const [showBox, setShowBox] = useState(true);

  const [refState, setRefState] = useState<ReferenceState>({ kind: "loading" });
  // Index of the reference image shown prominently (for the multi-image case).
  const [activeImageIndex, setActiveImageIndex] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  // Whether the reference panel (right) is open. Closed by default — opened via
  // its tab. The two side panels are mutually exclusive: each is locked to the
  // image's width, so opening both would overflow the viewport (3×44vw).
  const [showReference, setShowReference] = useState(false);
  // Whether the advanced-stats panel (left) is open. Closed by default.
  const [showStats, setShowStats] = useState(false);
  // Live rendered size of the detection image; the reference panel is locked
  // to these exact pixel dimensions so it always matches the image.
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  // Above `lg` the panels sit *beside* the image, locked to its pixel size;
  // below it they stack full-width beneath the image (a segmented control picks
  // which one shows), so nothing is crushed into ~44vw on a phone.
  const isDesktop = useIsDesktop();

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

  /** Correct the detection's species via the API, then notify the parent. */
  async function handleCorrect(chosen: string, isNew: boolean) {
    if (correctBusy) return;
    setCorrectBusy(true);
    setCorrectError(null);
    try {
      const updated = await api.detections.correct(id, chosen, isNew);
      onUpdate(updated);
      setCorrecting(false);
    } catch (e) {
      setCorrectError(e instanceof Error ? e.message : "Correction failed");
    } finally {
      setCorrectBusy(false);
    }
  }

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

  // Keyboard navigation. While the correction picker is open it owns the
  // keyboard (its own Esc cancels it, ↑/↓ move the list), so the lightbox's
  // Esc-to-close / arrow-to-navigate are suppressed to avoid double-handling.
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (correcting) return;
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && onPrev) onPrev();
      if (e.key === "ArrowRight" && onNext) onNext();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose, onPrev, onNext, correcting]);

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
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto overflow-x-hidden bg-ink/95 p-4 lg:items-center lg:overflow-hidden"
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

      {/* Image + reference row — stops click propagation so interacting inside
          doesn't close. On mobile it stacks (image, caption, then the chosen
          panel); on `lg`+ the panels flank the image. */}
      <div
        className="relative flex flex-col items-center lg:flex-row lg:items-start"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Advanced-stats panel (left) — mirrors the reference panel but folds
            to the left (animates margin-right). Locked to the image's exact
            rendered size, content scrolls internally. Mutually exclusive with the
            reference panel. Desktop only — on mobile it stacks below the image. ── */}
        {isDesktop && imgSize && (
          <div
            className="shrink-0 overflow-hidden transition-[width,margin-right,opacity] duration-300 ease-out motion-reduce:transition-none"
            style={{
              width: showStats ? imgSize.w : 0,
              marginRight: showStats ? "2.5rem" : 0,
              opacity: showStats ? 1 : 0,
              pointerEvents: showStats ? "auto" : "none",
            }}
            aria-hidden={!showStats}
          >
            <div
              className="overflow-y-auto rounded-lg border border-line bg-card shadow-plate-lift p-4"
              style={{ width: imgSize.w, height: imgSize.h }}
            >
              <h3 className="eyebrow mb-3">Advanced stats</h3>
              <AdvancedStatsPane detection={shown} />
            </div>
          </div>
        )}

        {/* ── Captured detection image (with the Field-guide tab on its edge) ──
            The column and its caption are locked to the image's exact rendered
            width (`imgSize.w`), and the image plate is `w-fit`, so every overlay
            — the top control bar, the box, and the Field-guide tab — anchors to
            the image itself. Nothing below can be wider than the image, so the
            plate is never pushed aside and the tab sits flush on its edge. */}
        <div className="flex flex-col items-start gap-3">
          <div className="relative w-fit">
            {mode === "video" && hasVideo ? (
              <video
                src={videoUrl}
                poster={thumbUrl}
                controls
                autoPlay
                loop
                muted
                // Lock the player to the still's already-measured rendered size.
                // A bare <video> collapses to its 300×150 intrinsic size until
                // metadata loads, then jumps to the real aspect ratio — shifting
                // the whole layout. The clip comes from the same main-stream frame
                // as the still, so imgSize matches its aspect ratio: pinning it
                // holds a stable footprint through loading and after. object-cover
                // keeps the square poster filling the box (no stretch) meanwhile.
                style={imgSize ? { width: imgSize.w, height: imgSize.h } : undefined}
                className="block max-h-[60vh] max-w-full rounded-lg bg-ink object-cover shadow-plate-lift lg:max-h-[80vh] lg:max-w-[44vw]"
              />
            ) : (
              <img
                // Keyed on the shown id so a fresh element mounts on every swap
                // and the develop-in animation replays. The bytes are already
                // cached (preloaded before the swap), so it resolves from a soft
                // blur straight into the sharp plate.
                key={id}
                ref={imgRef}
                src={fullUrl}
                alt={`Captured ${species}`}
                className="block max-h-[60vh] max-w-full animate-plate-develop rounded-lg bg-ink shadow-plate-lift lg:max-h-[80vh] lg:max-w-[44vw]"
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
                  left: `${shown.box_x! * 100}%`,
                  top: `${shown.box_y! * 100}%`,
                  width: `${shown.box_w! * 100}%`,
                  height: `${shown.box_h! * 100}%`,
                }}
                aria-hidden="true"
              />
            )}

            {/* ── On-image control bar ──
                The view controls (media + box) live *on* the image because they
                change what it shows; Close shares the bar. Grouped at the top
                under a soft scrim so the bird's usual lower-third stays clear and
                nothing collides with the native <video> control bar at the
                bottom. The scrim is click-through; only the controls take pointer
                events. */}
            <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between gap-2 rounded-t-lg bg-gradient-to-b from-ink/80 via-ink/35 to-transparent px-2.5 pb-12 pt-2.5">
              <div className="pointer-events-auto flex items-center gap-2">
                {/* Media (Photo / Video) — swaps the still for the clip */}
                <div
                  className="flex rounded-full bg-ink/55 p-0.5 ring-1 ring-paper/25 backdrop-blur"
                  role="group"
                  aria-label="Choose media"
                >
                  {(["photo", "video"] as const).map((m) => {
                    // The Video button stays visible even without a clip, but is
                    // disabled and explains why on hover. aria-disabled (not the
                    // native `disabled` attribute) keeps the title tooltip firing
                    // — browsers suppress hover events on natively-disabled
                    // buttons.
                    const unavailable = m === "video" && !hasVideo;
                    return (
                      <button
                        key={m}
                        className={`rounded-full px-3 py-1 text-xs font-medium capitalize transition-colors ${
                          mode === m ? "bg-gold text-ink" : "text-paper/85 hover:text-paper"
                        } ${unavailable ? "cursor-not-allowed opacity-40 hover:text-paper/85" : ""}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!unavailable) setMode(m);
                        }}
                        aria-pressed={mode === m}
                        aria-disabled={unavailable}
                        title={
                          unavailable
                            ? noVideoReasonText(shown.no_video_reason)
                            : undefined
                        }
                      >
                        {m}
                      </button>
                    );
                  })}
                </div>

                {/* Box on / off — only meaningful on the still */}
                {mode === "photo" && hasBox && (
                  <button
                    className={`rounded-full px-3 py-1.5 text-xs font-medium ring-1 backdrop-blur transition-colors ${
                      showBox
                        ? "bg-gold text-ink ring-gold"
                        : "bg-ink/55 text-paper ring-paper/25 hover:bg-ink/70"
                    }`}
                    onClick={(e) => { e.stopPropagation(); setShowBox((v) => !v); }}
                    aria-pressed={showBox}
                  >
                    {showBox ? "Box on" : "Box off"}
                  </button>
                )}
              </div>

              {/* Close */}
              <button
                className="pointer-events-auto rounded-full bg-ink/55 p-1.5 text-lg leading-none text-paper ring-1 ring-paper/25 backdrop-blur transition-colors hover:bg-ink/70"
                onClick={onClose}
                aria-label="Close"
              >
                ✕
              </button>
            </div>

            {/* Vertical Advanced-stats tab on the LEFT edge of the image —
                mirrors the Field-guide tab; opening it closes the reference
                panel (the two are mutually exclusive). */}
            <button
              className={`absolute right-full top-1/2 z-10 hidden -translate-y-1/2 rounded-l-lg px-2 py-3.5 text-[11px] font-semibold uppercase tracking-[0.15em] shadow-plate [writing-mode:vertical-rl] transition-colors lg:block ${
                showStats
                  ? "bg-gold text-ink"
                  : "bg-card/95 text-ink ring-1 ring-line hover:bg-card"
              }`}
              onClick={() => {
                setShowStats((v) => !v);
                setShowReference(false);
              }}
              aria-pressed={showStats}
              aria-label={showStats ? "Hide advanced stats" : "Show advanced stats"}
            >
              Advanced stats
            </button>

            {/* Vertical Field-guide tab on the right edge of the image */}
            <button
              className={`absolute left-full top-1/2 z-10 hidden -translate-y-1/2 rounded-r-lg px-2 py-3.5 text-[11px] font-semibold uppercase tracking-[0.15em] shadow-plate [writing-mode:vertical-rl] transition-colors lg:block ${
                showReference
                  ? "bg-gold text-ink"
                  : "bg-card/95 text-ink ring-1 ring-line hover:bg-card"
              }`}
              onClick={() => {
                setShowReference((v) => !v);
                setShowStats(false);
              }}
              aria-pressed={showReference}
              aria-label={showReference ? "Hide field guide" : "Show field guide"}
            >
              Field guide
            </button>
          </div>

          {/* ── Specimen label ──
              A compact caption card, locked to the image's exact width so it can
              never widen the plate. Reads like a guide's label: the species name
              and the margin-correction affordance up top, the readings on a quiet
              tabular line beneath, and the record actions (download / delete) on
              the right. */}
          <div
            className="w-full rounded-xl border border-line bg-card/95 px-4 py-3 shadow-plate"
            style={{ width: isDesktop && imgSize ? imgSize.w : undefined }}
          >
            <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-3">
              <div className="min-w-0">
                {/* Species + the margin-correction affordance (picker opens above) */}
                <div className="relative flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="font-display text-lg font-medium leading-tight text-ink">
                    {species}
                  </span>
                  <button
                    className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium text-bark transition-colors hover:bg-paper hover:text-ink"
                    onClick={(e) => {
                      e.stopPropagation();
                      setCorrecting((v) => !v);
                      setCorrectError(null);
                    }}
                    aria-expanded={correcting}
                    aria-label="Correct the species"
                    title="Wrong bird? Set the record straight."
                  >
                    <span aria-hidden="true">✎</span>
                    Correct ID
                  </button>
                  {correcting && (
                    <div className="absolute bottom-full left-0 z-20 mb-2">
                      <SpeciesPicker
                        current={species}
                        onConfirm={handleCorrect}
                        onCancel={() => {
                          setCorrecting(false);
                          setCorrectError(null);
                        }}
                        busy={correctBusy}
                        errorMessage={correctError}
                      />
                    </div>
                  )}
                </div>

                {/* Readings */}
                <div className="mt-1 flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5 text-xs">
                  {corrected ? (
                    <span className="flex items-baseline gap-1.5" title="Species set by you">
                      <span className="font-display text-sm italic text-gold-deep">
                        ✎ Corrected by you
                      </span>
                      {originalSpecies && (
                        <span className="text-[11px] text-bark">
                          model saw <span className="tnum">{confidencePct}%</span>{" "}
                          {originalSpecies}
                        </span>
                      )}
                    </span>
                  ) : (
                    <span
                      className="tnum font-medium text-gold-deep"
                      title="Species-classification confidence"
                    >
                      {confidencePct}% match
                    </span>
                  )}
                  {detectionPct !== null && (
                    <span
                      className="tnum text-bark"
                      title="Object-detection confidence (YOLO)"
                    >
                      {detectionPct}% spotted
                    </span>
                  )}
                  <span className="text-bark">{timeAgo(timestamp)}</span>
                </div>
              </div>

              {/* Record actions */}
              <div className="flex shrink-0 items-center gap-2">
                <a
                  href={mode === "video" && hasVideo ? videoUrl : fullUrl}
                  download
                  className="rounded-md border border-line bg-paper px-3 py-1.5 text-xs font-medium text-ink transition-colors hover:bg-card"
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
              </div>
            </div>
            {deleteError && <p className="mt-2 text-xs text-rust">{deleteError}</p>}
          </div>

          {/* ── Mobile panel switcher ──
              Below `lg` the two side panels can't flank the image, so they stack
              here as full-width sections. A segmented control (mirroring the
              History tabs) picks which one shows; picking the active one again
              closes it. Hidden on desktop, where the edge tabs + side panels
              take over. */}
          {!isDesktop && (
            <div className="w-full">
              <div
                className="flex gap-1 rounded-xl border border-line bg-card p-1"
                role="group"
                aria-label="More about this sighting"
              >
                <button
                  className={`flex-1 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                    showReference ? "bg-gold text-ink" : "text-bark hover:text-ink"
                  }`}
                  onClick={() => {
                    setShowReference((v) => !v);
                    setShowStats(false);
                  }}
                  aria-pressed={showReference}
                >
                  Field guide
                </button>
                <button
                  className={`flex-1 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                    showStats ? "bg-gold text-ink" : "text-bark hover:text-ink"
                  }`}
                  onClick={() => {
                    setShowStats((v) => !v);
                    setShowReference(false);
                  }}
                  aria-pressed={showStats}
                >
                  Advanced stats
                </button>
              </div>

              {showReference && (
                <div className="mt-3 rounded-lg border border-line bg-card p-4 shadow-plate">
                  <h3 className="eyebrow mb-3">Field guide</h3>
                  <ReferencePane
                    state={refState}
                    activeImageIndex={activeImageIndex}
                    onSelectImage={setActiveImageIndex}
                  />
                </div>
              )}

              {showStats && (
                <div className="mt-3 rounded-lg border border-line bg-card p-4 shadow-plate">
                  <h3 className="eyebrow mb-3">Advanced stats</h3>
                  <AdvancedStatsPane detection={shown} />
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Reference panel — locked to the image's exact rendered size ──
            Stays mounted (once the image size is known) so it can unfold and
            fold away smoothly: the outer wrapper animates its width, left
            margin (the gap to the image) and opacity, clipping the fixed-size
            inner card so its content never reflows mid-animation. The image is
            centred in the row, so it glides aside as the panel grows. The
            global prefers-reduced-motion guard zeroes these durations. Desktop
            only — on mobile the guide stacks below the image. */}
        {isDesktop && imgSize && (
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
