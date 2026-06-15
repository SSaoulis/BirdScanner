import { useEffect } from "react";
import { api, timeAgo, type Detection } from "../api";

interface LightboxProps {
  /** The detection currently displayed in the lightbox. */
  detection: Detection;
  /** Called when the lightbox should close. */
  onClose: () => void;
  /** Called to navigate to the previous detection. Omit or pass null when at the start. */
  onPrev: (() => void) | null;
  /** Called to navigate to the next detection. Omit or pass null when at the end. */
  onNext: (() => void) | null;
}

/**
 * Full-screen modal overlay that loads the full-resolution image for a
 * detection on open. Supports keyboard navigation (Esc, ArrowLeft,
 * ArrowRight) and prev/next arrow buttons.
 */
export function Lightbox({ detection, onClose, onPrev, onNext }: LightboxProps) {
  const { id, species, confidence, timestamp } = detection;
  const fullUrl = api.images.fullUrl(id);
  const confidencePct = (confidence * 100).toFixed(1);

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

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`Full image of ${species}`}
    >
      {/* Prev arrow */}
      {onPrev && (
        <button
          className="absolute left-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-slate-800/80 hover:bg-slate-700 text-white text-2xl transition-colors z-10"
          onClick={(e) => { e.stopPropagation(); onPrev(); }}
          aria-label="Previous image"
        >
          &#8592;
        </button>
      )}

      {/* Image container — stops click propagation so clicking the image itself doesn't close */}
      <div
        className="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center gap-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          className="absolute -top-3 -right-3 z-10 p-1.5 rounded-full bg-slate-800/80 hover:bg-slate-700 text-white text-lg leading-none"
          onClick={onClose}
          aria-label="Close lightbox"
        >
          ✕
        </button>

        <img
          src={fullUrl}
          alt={species}
          className="max-w-full max-h-[75vh] object-contain rounded-lg shadow-2xl"
        />

        {/* Caption bar */}
        <div className="flex items-center gap-4 px-4 py-2 bg-slate-800/90 rounded-xl text-sm">
          <span className="font-semibold text-white">{species}</span>
          <span className="text-emerald-400 font-mono">{confidencePct}%</span>
          <span className="text-slate-400">{timeAgo(timestamp)}</span>
          <a
            href={fullUrl}
            download
            className="ml-2 text-slate-400 hover:text-white underline"
            onClick={(e) => e.stopPropagation()}
          >
            Download
          </a>
        </div>
      </div>

      {/* Next arrow */}
      {onNext && (
        <button
          className="absolute right-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-slate-800/80 hover:bg-slate-700 text-white text-2xl transition-colors z-10"
          onClick={(e) => { e.stopPropagation(); onNext(); }}
          aria-label="Next image"
        >
          &#8594;
        </button>
      )}
    </div>
  );
}
