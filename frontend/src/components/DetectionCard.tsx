import { api, timeAgo, type Detection } from "../api";

interface DetectionCardProps {
  /** The detection to render. */
  detection: Detection;
  /**
   * When provided, renders the card in selectable mode. Clicking the card
   * calls this handler instead of navigating to the full-res image.
   * A checkbox overlay and selection ring are shown.
   */
  onSelect?: (detection: Detection) => void;
  /** Whether this card is currently selected (only meaningful when onSelect is provided). */
  selected?: boolean;
  /**
   * When provided, clicking the image thumbnail opens the lightbox rather
   * than navigating away. Has no effect when onSelect is also provided.
   */
  onOpenLightbox?: (detection: Detection) => void;
}

/**
 * A single detection rendered as a field-guide "specimen plate": the captured
 * photo mounted above a ruled caption bearing the species (in the journal
 * serif), how sure the match is, and how long ago it was spotted.
 *
 * Supports three interaction modes:
 * - Default: wraps the image in a link to the full-res photo.
 * - Lightbox: calls onOpenLightbox on click.
 * - Select: calls onSelect on click; shows a checkbox + selection ring.
 */
export function DetectionCard({ detection, onSelect, selected, onOpenLightbox }: DetectionCardProps) {
  const { id, species, confidence, detection_confidence, timestamp } = detection;
  const thumbnailUrl = api.images.thumbnailUrl(id);
  const confidencePct = (confidence * 100).toFixed(0);
  const detectionPct =
    detection_confidence != null ? (detection_confidence * 100).toFixed(0) : null;

  const isSelectable = Boolean(onSelect);

  function handleClick(e: React.MouseEvent) {
    if (onSelect) {
      e.preventDefault();
      onSelect(detection);
    }
  }

  const ringClass =
    isSelectable && selected
      ? "ring-2 ring-gold border-gold"
      : "border-line hover:-translate-y-0.5 hover:shadow-plate-lift";

  const imageContent = (
    <>
      <div className={`relative w-full h-36 bg-paper ${isSelectable ? "cursor-pointer" : ""}`}>
        <img
          src={thumbnailUrl}
          alt={species}
          className="w-full h-full object-cover"
          loading="lazy"
        />
        {/* Checkbox overlay (selectable mode) */}
        {isSelectable && (
          <div className="absolute top-2 left-2">
            <div
              className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                selected
                  ? "bg-gold border-gold"
                  : "bg-card/80 border-line"
              }`}
            >
              {selected && (
                <svg className="w-3 h-3 text-card" viewBox="0 0 12 12" fill="none">
                  <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </div>
          </div>
        )}
        {/* Lightbox trigger (non-selectable mode: full-image hover overlay) */}
        {onOpenLightbox && !isSelectable && (
          <button
            className="absolute inset-0 w-full h-full opacity-0 group-hover:opacity-100 flex items-center justify-center bg-ink/35 transition-opacity duration-200"
            onClick={(e) => { e.preventDefault(); onOpenLightbox(detection); }}
            aria-label={`Take a closer look at ${species}`}
          >
            <span className="rounded-full bg-paper px-3 py-1 text-xs font-semibold text-ink shadow-plate">
              Closer look
            </span>
          </button>
        )}
        {/* Lightbox trigger (selectable mode: small icon in top-right corner) */}
        {onOpenLightbox && isSelectable && (
          <button
            className="absolute top-2 right-2 p-1 rounded bg-card/85 hover:bg-card text-ink opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10 shadow-plate"
            onClick={(e) => { e.stopPropagation(); e.preventDefault(); onOpenLightbox(detection); }}
            aria-label={`Take a closer look at ${species}`}
            tabIndex={-1}
          >
            {/* Expand / view icon */}
            <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 2h4M2 2v4M14 2h-4M14 2v4M2 14h4M2 14v-4M14 14h-4M14 14v-4" strokeLinecap="round" />
            </svg>
          </button>
        )}
      </div>
      {/* Caption plate — ruled off from the photo like a guide's specimen label */}
      <div className="border-t border-line px-3 py-2">
        <p className="font-display text-[0.95rem] font-medium leading-tight text-ink truncate" title={species}>
          {species}
        </p>
        <div className="mt-1 flex items-center justify-between text-xs">
          <span className="flex items-center gap-1.5">
            <span className="tnum font-medium text-gold-deep" title="Species-classification confidence">
              {confidencePct}% match
            </span>
            {detectionPct !== null && (
              <span className="tnum text-bark" title="Object-detection confidence (YOLO)">
                · {detectionPct}% spotted
              </span>
            )}
          </span>
          <span className="text-bark">{timeAgo(timestamp)}</span>
        </div>
      </div>
    </>
  );

  const baseClass =
    "flex-shrink-0 w-44 overflow-hidden rounded-xl border bg-card shadow-plate transition-all duration-200 group";

  if (isSelectable) {
    return (
      <div
        className={`${baseClass} cursor-pointer ${ringClass}`}
        onClick={handleClick}
        role="checkbox"
        aria-checked={selected ?? false}
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); onSelect!(detection); } }}
      >
        {imageContent}
      </div>
    );
  }

  return (
    <a
      href={api.images.fullUrl(id)}
      target="_blank"
      rel="noopener noreferrer"
      className={`${baseClass} ${ringClass}`}
      onClick={onOpenLightbox ? (e) => { e.preventDefault(); onOpenLightbox(detection); } : undefined}
    >
      {imageContent}
    </a>
  );
}
