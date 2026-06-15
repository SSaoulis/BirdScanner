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
 * Card component that renders a detection thumbnail, species label,
 * confidence badge, and timestamp. Supports three interaction modes:
 * - Default: wraps image in a link to full-res.
 * - Lightbox: calls onOpenLightbox on click.
 * - Select: calls onSelect on click; shows checkbox + selection ring.
 */
export function DetectionCard({ detection, onSelect, selected, onOpenLightbox }: DetectionCardProps) {
  const { id, species, confidence, timestamp } = detection;
  const thumbnailUrl = api.images.thumbnailUrl(id);
  const confidencePct = (confidence * 100).toFixed(1);

  const isSelectable = Boolean(onSelect);

  function handleClick(e: React.MouseEvent) {
    if (onSelect) {
      e.preventDefault();
      onSelect(detection);
    }
  }

  const ringClass = isSelectable && selected
    ? "ring-2 ring-emerald-400"
    : isSelectable
    ? "hover:ring-2 hover:ring-slate-500"
    : "hover:ring-2 hover:ring-emerald-500";

  const imageContent = (
    <>
      <div className={`relative w-full h-36 bg-slate-900 ${isSelectable ? "cursor-pointer" : ""}`}>
        <img
          src={thumbnailUrl}
          alt={species}
          className="w-full h-full object-cover group-hover:opacity-90 transition-opacity duration-150"
          loading="lazy"
        />
        {/* Checkbox overlay (selectable mode) */}
        {isSelectable && (
          <div className="absolute top-2 left-2">
            <div
              className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                selected
                  ? "bg-emerald-500 border-emerald-500"
                  : "bg-slate-900/60 border-slate-400"
              }`}
            >
              {selected && (
                <svg className="w-3 h-3 text-white" viewBox="0 0 12 12" fill="none">
                  <path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </div>
          </div>
        )}
        {/* Lightbox trigger (non-selectable) */}
        {onOpenLightbox && !isSelectable && (
          <button
            className="absolute inset-0 w-full h-full opacity-0 group-hover:opacity-100 flex items-center justify-center bg-black/40 transition-opacity duration-150"
            onClick={(e) => { e.preventDefault(); onOpenLightbox(detection); }}
            aria-label={`Open lightbox for ${species}`}
          >
            <span className="text-white text-xs font-semibold bg-slate-800/80 px-2 py-1 rounded">View</span>
          </button>
        )}
      </div>
      <div className="p-2 space-y-0.5">
        <p className="text-sm font-semibold text-white truncate" title={species}>
          {species}
        </p>
        <div className="flex items-center justify-between">
          <span className="text-xs text-emerald-400 font-mono">{confidencePct}%</span>
          <span className="text-xs text-slate-500">{timeAgo(timestamp)}</span>
        </div>
      </div>
    </>
  );

  if (isSelectable) {
    return (
      <div
        className={`flex-shrink-0 w-44 bg-slate-800 rounded-xl overflow-hidden transition-all duration-150 group cursor-pointer ${ringClass}`}
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
      className={`flex-shrink-0 w-44 bg-slate-800 rounded-xl overflow-hidden transition-all duration-150 group ${ringClass}`}
      onClick={onOpenLightbox ? (e) => { e.preventDefault(); onOpenLightbox(detection); } : undefined}
    >
      {imageContent}
    </a>
  );
}
