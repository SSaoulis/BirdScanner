import type { ReactNode } from "react";
import { timeAgo } from "../api";

/**
 * Human-readable explanation for why a sighting has no video clip, shown as the
 * tooltip on the disabled Video toggle. Mirrors the `no_video_reason` values the
 * detector persists (see `birdscanner/ml/classification_pipeline.py`).
 */
export function noVideoReasonText(reason: string | null): string {
  switch (reason) {
    case "recorder_busy":
      return "No clip — the recorder was busy saving another sighting's video. Only one clip records at a time to spare the Pi's CPU, so this sighting overlapped another recording.";
    case "disabled":
      return "No clip — video recording is turned off.";
    default:
      return "No clip available for this sighting.";
  }
}

interface PlateControlBarProps {
  /** Position of this sighting within the parent list, for the mobile chip. */
  position?: { index: number; total: number } | null;
  /** Which media the plate is showing (drives the Photo/Video pressed state). */
  mode: "photo" | "video";
  /** Whether a clip exists (the Video pill is disabled + explained when false). */
  hasVideo: boolean;
  /** Whether the sighting has a persisted box (gates the Box toggle). */
  hasBox: boolean;
  /** Whether the box overlay is currently shown. */
  showBox: boolean;
  /** The sighting's `no_video_reason`, for the disabled-Video tooltip. */
  noVideoReason: string | null;
  /**
   * When false the bar is a static, non-interactive mirror (used by the swipe
   * neighbour preview): buttons render but take no pointer events and fire no
   * handlers. Defaults to true (the live card).
   */
  interactive?: boolean;
  /** Select the Photo/Video media. Omit for a static bar. */
  onSelectMode?: (mode: "photo" | "video") => void;
  /** Toggle the box overlay. Omit for a static bar. */
  onToggleBox?: () => void;
  /** Close the lightbox. Omit for a static bar. */
  onClose?: () => void;
}

/**
 * The on-image top control bar (media Photo/Video toggle, box on/off, Close, and
 * the mobile position chip), grouped at the top under a soft scrim so the bird's
 * lower-third stays clear. Shared by the live lightbox card and the swipe
 * neighbour preview so the controls look identical while a plate slides in;
 * `interactive={false}` renders the same chrome inert.
 */
export function PlateControlBar({
  position,
  mode,
  hasVideo,
  hasBox,
  showBox,
  noVideoReason,
  interactive = true,
  onSelectMode,
  onToggleBox,
  onClose,
}: PlateControlBarProps) {
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between gap-2 rounded-t-lg bg-gradient-to-b from-ink/80 via-ink/35 to-transparent px-2.5 pb-12 pt-2.5">
      {/* Mobile position chip — the affordance for swipe navigation.
          Non-interactive; hidden on desktop, where the edge arrows show. */}
      {position && (
        <span className="tnum pointer-events-none absolute left-1/2 top-2.5 -translate-x-1/2 rounded-full bg-ink/55 px-2.5 py-1 text-xs font-medium text-paper ring-1 ring-paper/25 backdrop-blur lg:hidden">
          {position.index + 1} / {position.total}
        </span>
      )}
      <div className={`flex items-center gap-2 ${interactive ? "pointer-events-auto" : ""}`}>
        {/* Media (Photo / Video) — swaps the still for the clip */}
        <div
          className="flex rounded-full bg-ink/55 p-0.5 ring-1 ring-paper/25 backdrop-blur"
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
                className={`rounded-full px-3 py-1 text-xs font-medium capitalize transition-colors ${
                  mode === m ? "bg-gold text-ink" : "text-paper/85 hover:text-paper"
                } ${unavailable ? "cursor-not-allowed opacity-40 hover:text-paper/85" : ""}`}
                onClick={
                  interactive && onSelectMode
                    ? (e) => {
                        e.stopPropagation();
                        if (!unavailable) onSelectMode(m);
                      }
                    : undefined
                }
                aria-pressed={mode === m}
                aria-disabled={unavailable}
                title={unavailable ? noVideoReasonText(noVideoReason) : undefined}
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
            onClick={
              interactive && onToggleBox
                ? (e) => {
                    e.stopPropagation();
                    onToggleBox();
                  }
                : undefined
            }
            aria-pressed={showBox}
          >
            {showBox ? "Box on" : "Box off"}
          </button>
        )}
      </div>

      {/* Close */}
      <button
        className={`rounded-full bg-ink/55 p-1.5 text-lg leading-none text-paper ring-1 ring-paper/25 backdrop-blur transition-colors hover:bg-ink/70 ${
          interactive ? "pointer-events-auto" : ""
        }`}
        onClick={interactive ? onClose : undefined}
        aria-label="Close"
      >
        ✕
      </button>
    </div>
  );
}

interface SpecimenLabelProps {
  /** The (possibly corrected) species name. */
  species: string;
  /** Classification confidence, pre-formatted as a percentage string. */
  confidencePct: string;
  /** Object-detection (YOLO) confidence as a percentage string, or null. */
  detectionPct: string | null;
  /** Whether the species was corrected by the user. */
  corrected: boolean;
  /** The model's original pick, shown when `corrected`. */
  originalSpecies: string | null;
  /** ISO timestamp of the sighting, rendered via `timeAgo`. */
  timestamp: string;
  /** Href for the Download action (the still or the clip). */
  downloadUrl: string;
  /** Fixed pixel width, locking the caption to the image size (desktop). */
  width?: number;
  /**
   * When false the caption is a static, non-interactive mirror (swipe neighbour
   * preview): the Correct-ID / Download / Delete controls render but do nothing.
   * Defaults to true (the live card).
   */
  interactive?: boolean;
  /** Whether the correction picker is open (drives Correct-ID's pressed state). */
  correcting?: boolean;
  /** Toggle the correction picker. Omit for a static caption. */
  onToggleCorrect?: () => void;
  /** Whether a delete is in flight. */
  deleting?: boolean;
  /** Delete the sighting. Omit for a static caption. */
  onDelete?: () => void;
  /** Error message from a failed delete, shown beneath the actions. */
  deleteError?: string | null;
  /** The correction picker popover, positioned above the species row when open. */
  picker?: ReactNode;
}

/**
 * The specimen-label caption card: the species name and the correction
 * affordance up top, the readings on a quiet tabular line, and the record
 * actions (Download / Delete) on the right. Shared by the live lightbox card and
 * the swipe neighbour preview; `interactive={false}` renders it inert so the
 * caption looks identical while a plate slides in.
 */
export function SpecimenLabel({
  species,
  confidencePct,
  detectionPct,
  corrected,
  originalSpecies,
  timestamp,
  downloadUrl,
  width,
  interactive = true,
  correcting = false,
  onToggleCorrect,
  deleting = false,
  onDelete,
  deleteError,
  picker,
}: SpecimenLabelProps) {
  return (
    <div
      className="w-full rounded-xl border border-line bg-card/95 px-4 py-3 shadow-plate"
      style={{ width }}
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
              onClick={
                interactive && onToggleCorrect
                  ? (e) => {
                      e.stopPropagation();
                      onToggleCorrect();
                    }
                  : undefined
              }
              aria-expanded={correcting}
              aria-label="Correct the species"
              title="Wrong bird? Set the record straight."
            >
              <span aria-hidden="true">✎</span>
              Correct ID
            </button>
            {picker && <div className="absolute bottom-full left-0 z-20 mb-2">{picker}</div>}
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
                    model saw <span className="tnum">{confidencePct}%</span> {originalSpecies}
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
              <span className="tnum text-bark" title="Object-detection confidence (YOLO)">
                {detectionPct}% spotted
              </span>
            )}
            <span className="text-bark">{timeAgo(timestamp)}</span>
          </div>
        </div>

        {/* Record actions */}
        <div className="flex shrink-0 items-center gap-2">
          <a
            href={downloadUrl}
            download
            className="rounded-md border border-line bg-paper px-3 py-1.5 text-xs font-medium text-ink transition-colors hover:bg-card"
            onClick={interactive ? (e) => e.stopPropagation() : undefined}
          >
            Download
          </a>
          <button
            className="rounded-md bg-rust px-3 py-1.5 text-xs font-medium text-card transition-colors hover:brightness-110 disabled:opacity-50"
            onClick={
              interactive && onDelete
                ? (e) => {
                    e.stopPropagation();
                    onDelete();
                  }
                : undefined
            }
            disabled={deleting}
          >
            {deleting ? "Deleting…" : "Delete"}
          </button>
        </div>
      </div>
      {deleteError && <p className="mt-2 text-xs text-rust">{deleteError}</p>}
    </div>
  );
}

interface MobilePanelTabsProps {
  /** Whether the Field-guide panel is open (pressed state). */
  showReference: boolean;
  /** Whether the Advanced-stats panel is open (pressed state). */
  showStats: boolean;
  /**
   * When false the segmented control is inert (swipe neighbour preview).
   * Defaults to true (the live card).
   */
  interactive?: boolean;
  /** Toggle the Field-guide panel. Omit for a static control. */
  onSelectReference?: () => void;
  /** Toggle the Advanced-stats panel. Omit for a static control. */
  onSelectStats?: () => void;
}

/**
 * The mobile Field-guide / Advanced-stats segmented control (below `lg`, where
 * the side panels can't flank the image). Just the tab bar — the panels
 * themselves stay in the live card. Shared with the neighbour preview so the
 * switcher is visible while a plate slides in; `interactive={false}` inert.
 */
export function MobilePanelTabs({
  showReference,
  showStats,
  interactive = true,
  onSelectReference,
  onSelectStats,
}: MobilePanelTabsProps) {
  return (
    <div
      className="flex gap-1 rounded-xl border border-line bg-card p-1"
      role="group"
      aria-label="More about this sighting"
    >
      <button
        className={`flex-1 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
          showReference ? "bg-gold text-ink" : "text-bark hover:text-ink"
        }`}
        onClick={interactive ? onSelectReference : undefined}
        aria-pressed={showReference}
      >
        Field guide
      </button>
      <button
        className={`flex-1 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
          showStats ? "bg-gold text-ink" : "text-bark hover:text-ink"
        }`}
        onClick={interactive ? onSelectStats : undefined}
        aria-pressed={showStats}
      >
        Advanced stats
      </button>
    </div>
  );
}
