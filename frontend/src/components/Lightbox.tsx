import { Suspense, lazy, useEffect, useRef, useState } from "react";
import { api, ApiError, type Detection, type SpeciesReference } from "../api";
import { SpeciesPicker } from "./SpeciesPicker";
import { AdvancedStatsPane } from "./AdvancedStats";
import { PlateControlBar, SpecimenLabel, MobilePanelTabs } from "./LightboxChrome";
import { useIsDesktop } from "../hooks/useMediaQuery";

// The mobile swipe filmstrip pulls in `motion`; lazy-load it so desktop and the
// initial bundle never pay for it. It only mounts once the lightbox is open on
// a mobile viewport (see the mobile branch below).
const MobileSwipeStrip = lazy(() => import("./MobileSwipeStrip"));

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
  /**
   * Position of this detection within the parent's list, for the mobile counter
   * chip. `index` is 0-based. Omit to hide the chip (navigation still works).
   */
  position?: { index: number; total: number } | null;
  /**
   * The neighbouring sightings, used to render the mobile swipe filmstrip so a
   * drag slides the real neighbour plate into frame. Pass null at a list end
   * (mirrors `onPrev`/`onNext`); the index change still funnels through those.
   */
  prevDetection?: Detection | null;
  nextDetection?: Detection | null;
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
 *
 * On mobile the floating arrows give way to a physical, finger-tracking swipe:
 * the card is the centre slot of a three-up filmstrip (prev / current / next)
 * that follows the drag, committing to a neighbour past a distance/velocity
 * threshold and rubber-banding back otherwise (see `MobileSwipeStrip`).
 */
export function Lightbox({
  detection,
  onClose,
  onPrev,
  onNext,
  onDelete,
  onUpdate,
  position,
  prevDetection,
  nextDetection,
}: LightboxProps) {
  // The detection whose plate is *currently on screen*. Derived straight from
  // the prop (no lag) so that on a committed mobile swipe the centre card, the
  // neighbour slots and the recentre trigger all update on the *same* render —
  // a one-render-stale copy would flash the wrong plate as the filmstrip snaps
  // back to centre. The incoming `detection` covers both a different sighting
  // (prev/next) and an in-place species correction (same id); the breathing
  // blur-up loader covers the full-res fetch, so nothing needs holding back.
  // Everything below derives from `shown`.
  const shown = detection;

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
  // Whether the active media has finished loading. Drives the breathing blur-up
  // loader: false while the full-res still / clip fetches, flipped true by the
  // media element's own load event so the loader can fade out. Reset on every
  // sighting change so navigating shows the loader again for the new record.
  const [photoReady, setPhotoReady] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  // Whether the species-correction picker is open, plus its in-flight/error state.
  const [correcting, setCorrecting] = useState(false);
  const [correctBusy, setCorrectBusy] = useState(false);
  const [correctError, setCorrectError] = useState<string | null>(null);
  // Live rendered size of the detection image; the reference panel is locked
  // to these exact pixel dimensions so it always matches the image.
  const imgRef = useRef<HTMLImageElement | null>(null);
  useEffect(() => {
    setMode("photo");
    // Reset load-readiness for the new record so the blur-up loader covers its
    // fetch — unless the freshly-mounted still is already cached (e.g. a mobile
    // neighbour preview preloaded it during a swipe), in which case treat it as
    // ready so the loader never flashes over an image we already have.
    const el = imgRef.current;
    const cached = el !== null && el.complete && el.naturalWidth > 0;
    setPhotoReady(cached);
    setVideoReady(false);
    // Navigating to another sighting closes the picker so it never lingers over
    // the wrong record.
    setCorrecting(false);
    setCorrectBusy(false);
    setCorrectError(null);
  }, [id]);
  const confidencePct = (confidence * 100).toFixed(1);
  const detectionPct =
    detection_confidence != null ? (detection_confidence * 100).toFixed(1) : null;
  // Whether the media the plate is actually showing has finished loading. Drives
  // the breathing loader's fade-out and the loading footprint lock.
  const activeReady = mode === "video" && hasVideo ? videoReady : photoReady;

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
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null);
  // Above `lg` the panels sit *beside* the image, locked to its pixel size;
  // below it they stack full-width beneath the image (a segmented control picks
  // which one shows), so nothing is crushed into ~44vw on a phone. Mobile also
  // trades the floating arrows for the finger-tracking swipe filmstrip.
  const isDesktop = useIsDesktop();

  // Positions of the swipe neighbours, so their preview chips read the right
  // "n / total" as they slide in (and don't flicker the number on commit). Null
  // at a list end, mirroring prevDetection/nextDetection.
  const prevPosition =
    position && position.index > 0
      ? { index: position.index - 1, total: position.total }
      : null;
  const nextPosition =
    position && position.index < position.total - 1
      ? { index: position.index + 1, total: position.total }
      : null;

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

  // ── The interactive detection card ──
  // The image plate (with its overlays), the specimen-label caption and — on
  // mobile — the panel switcher. Shared by the desktop side-panel row and the
  // mobile swipe filmstrip (as its live centre slot), so it is built once here.
  const card = (
    <div className="flex flex-col items-start gap-3">
      {/* While the active media is still loading, pin the plate to the
          last-known rendered size so the footprint (and the caption / side
          panels below) doesn't collapse and jump. `imgSize` still holds the
          previous sighting's size during the swap; the breathing blur-up
          loader fills it. Once the media is ready we drop the fixed size and
          revert to `w-fit` wrapping the media at its natural size. */}
      <div
        className="relative w-fit"
        style={
          !activeReady && imgSize
            ? { width: imgSize.w, height: imgSize.h }
            : undefined
        }
      >
        {mode === "video" && hasVideo ? (
          <video
            src={videoUrl}
            poster={thumbUrl}
            controls
            autoPlay
            loop
            muted
            onCanPlay={() => setVideoReady(true)}
            onWaiting={() => setVideoReady(false)}
            onError={() => setVideoReady(true)}
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
            // Keyed on the shown id so a fresh element mounts on every swap.
            // The plate is held hidden (opacity-0) while the full-res bytes
            // network-fetch, so you never watch it paint in top-to-bottom
            // beneath the blur loader (below, which covers the gap). Once
            // loaded, onLoad flips `photoReady`: on desktop the develop-in
            // reveal replays as the loader fades; on mobile the develop is
            // dropped because the physical filmstrip slide is already the
            // transition (a blur re-develop under it would double up).
            key={id}
            ref={imgRef}
            src={fullUrl}
            alt={`Captured ${species}`}
            onLoad={() => setPhotoReady(true)}
            onError={() => setPhotoReady(true)}
            className={`block max-h-[60vh] max-w-full rounded-lg bg-ink shadow-plate-lift lg:max-h-[80vh] lg:max-w-[44vw] ${
              photoReady ? (isDesktop ? "animate-plate-develop" : "") : "opacity-0"
            }`}
          />
        )}

        {/* ── Breathing blur-up loader ──
            A blurred, gently breathing copy of the low-res thumbnail shown
            over the plate while the full-res still / clip loads, so a swap
            never reads as frozen. An outer wrapper fades it out (opacity)
            once the media is ready; the inner thumbnail carries the breathing
            animation (kept separate so the fade-out and the pulse don't fight
            over `opacity`). Click-through and hidden from assistive tech. */}
        <div
          className="pointer-events-none absolute inset-0 z-[5] overflow-hidden rounded-lg transition-opacity duration-500 ease-out"
          style={{ opacity: activeReady ? 0 : 1 }}
          aria-hidden="true"
        >
          <img
            key={id}
            src={thumbUrl}
            alt=""
            // The breathing (blur) animation only runs while loading; once
            // ready we drop it (keeping a static blur for the opacity
            // fade-out) so an infinite filter animation isn't left
            // compositing behind every open lightbox.
            className={`h-full w-full rounded-lg bg-ink object-cover blur-[8px] ${
              activeReady ? "" : "animate-plate-breathe"
            }`}
          />
        </div>

        {/* Detection box overlay — positioned in normalized [0,1] space over
            the rendered image, so it scales with whatever size the image is
            capped to. Only meaningful on the still, so it is hidden in video
            mode, when toggled off, or for legacy boxless rows. Gated on
            `photoReady` so the box fades in with the sharp plate rather than
            sitting over the blurred loader. */}
        {mode === "photo" && hasBox && showBox && photoReady && (
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
            change what it shows; Close shares the bar. Shared with the swipe
            neighbour preview (see `LightboxChrome`) so the chrome is visible as
            a plate slides in rather than popping in after the swipe commits. */}
        <PlateControlBar
          position={position}
          mode={mode}
          hasVideo={hasVideo}
          hasBox={hasBox}
          showBox={showBox}
          noVideoReason={shown.no_video_reason}
          onSelectMode={setMode}
          onToggleBox={() => setShowBox((v) => !v)}
          onClose={onClose}
        />

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
          never widen the plate. Shared with the swipe neighbour preview (see
          `LightboxChrome`); the live card wires the correction picker + record
          actions, the preview renders the same caption inert. */}
      <SpecimenLabel
        species={species}
        confidencePct={confidencePct}
        detectionPct={detectionPct}
        corrected={corrected}
        originalSpecies={originalSpecies}
        timestamp={timestamp}
        downloadUrl={mode === "video" && hasVideo ? videoUrl : fullUrl}
        width={isDesktop && imgSize ? imgSize.w : undefined}
        correcting={correcting}
        onToggleCorrect={() => {
          setCorrecting((v) => !v);
          setCorrectError(null);
        }}
        deleting={deleting}
        onDelete={handleDelete}
        deleteError={deleteError}
        picker={
          correcting ? (
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
          ) : null
        }
      />

      {/* ── Mobile panel switcher ──
          Below `lg` the two side panels can't flank the image, so they stack
          here as full-width sections. A segmented control (mirroring the
          History tabs) picks which one shows; picking the active one again
          closes it. Hidden on desktop, where the edge tabs + side panels
          take over. */}
      {!isDesktop && (
        <div className="w-full">
          <MobilePanelTabs
            showReference={showReference}
            showStats={showStats}
            onSelectReference={() => {
              setShowReference((v) => !v);
              setShowStats(false);
            }}
            onSelectStats={() => {
              setShowStats((v) => !v);
              setShowReference(false);
            }}
          />

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
  );

  // ── Desktop-only side panels ──
  // Locked to the image's exact rendered size, folding open/closed. Only built
  // once the image size is known; rendered solely in the desktop row.
  const statsPanel = imgSize ? (
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
  ) : null;

  const referencePanel = imgSize ? (
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
  ) : null;

  return (
    <div
      className={
        isDesktop
          ? "fixed inset-0 z-50 flex items-center justify-center overflow-hidden bg-ink/95 p-4"
          : "fixed inset-0 z-50 overflow-y-auto overflow-x-hidden bg-ink/95"
      }
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`A closer look at ${species}`}
    >
      {/* Prev arrow — desktop only; mobile uses the swipe filmstrip. */}
      {onPrev && (
        <button
          className="hidden lg:block absolute left-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-card/90 hover:bg-card text-ink text-2xl shadow-plate transition-colors z-10"
          onClick={(e) => { e.stopPropagation(); onPrev(); }}
          aria-label="Previous sighting"
        >
          &#8592;
        </button>
      )}

      {isDesktop ? (
        // Desktop: the panels flank the image. Stops click propagation so
        // interacting inside doesn't close.
        <div
          className="relative flex flex-row items-start"
          onClick={(e) => e.stopPropagation()}
        >
          {statsPanel}
          {card}
          {referencePanel}
        </div>
      ) : (
        // Mobile: the card is the live centre slot of a finger-tracking
        // three-up filmstrip that slides in the neighbouring plates. While the
        // (lazy) strip chunk loads, the card shows centred without swipe.
        <Suspense
          fallback={
            <div
              className="flex w-full items-start justify-center px-4 py-4"
              onClick={(e) => e.stopPropagation()}
            >
              {card}
            </div>
          }
        >
          <MobileSwipeStrip
            currentId={id}
            prevDetection={prevDetection ?? null}
            nextDetection={nextDetection ?? null}
            prevPosition={prevPosition}
            nextPosition={nextPosition}
            onPrev={onPrev}
            onNext={onNext}
            enabled={mode !== "video" && !correcting}
          >
            {card}
          </MobileSwipeStrip>
        </Suspense>
      )}

      {/* Next arrow — desktop only; mobile uses the swipe filmstrip. */}
      {onNext && (
        <button
          className="hidden lg:block absolute right-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-card/90 hover:bg-card text-ink text-2xl shadow-plate transition-colors z-10"
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
