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
  const { id, species, confidence, timestamp } = detection;
  const fullUrl = api.images.fullUrl(id);
  const confidencePct = (confidence * 100).toFixed(1);

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
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={`Comparison view for ${species}`}
    >
      {/* Prev arrow */}
      {onPrev && (
        <button
          className="absolute left-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-slate-800/80 hover:bg-slate-700 text-white text-2xl transition-colors z-10"
          onClick={(e) => { e.stopPropagation(); onPrev(); }}
          aria-label="Previous detection"
        >
          &#8592;
        </button>
      )}

      {/* Image + reference row — stops click propagation so interacting inside doesn't close */}
      <div
        className="relative flex items-start gap-10"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Captured detection image (with the Reference tab on its edge) ── */}
        <div className="flex flex-col gap-3">
          <div className="relative">
            <img
              ref={imgRef}
              src={fullUrl}
              alt={`Captured ${species}`}
              className="block max-h-[80vh] max-w-[44vw] rounded-lg bg-black shadow-2xl"
            />

            {/* Close button */}
            <button
              className="absolute top-2 right-2 z-10 p-1.5 rounded-full bg-slate-800/80 hover:bg-slate-700 text-white text-lg leading-none"
              onClick={onClose}
              aria-label="Close lightbox"
            >
              ✕
            </button>

            {/* Vertical Reference tab on the right edge of the image */}
            <button
              className={`absolute top-1/2 left-full -translate-y-1/2 px-1.5 py-3 rounded-r-lg text-xs font-semibold uppercase tracking-wide [writing-mode:vertical-rl] transition-colors ${
                showReference
                  ? "bg-emerald-600 text-white"
                  : "bg-slate-800/90 text-slate-200 hover:bg-slate-700"
              }`}
              onClick={() => setShowReference((v) => !v)}
              aria-pressed={showReference}
              aria-label={showReference ? "Hide species reference" : "Show species reference"}
            >
              Reference
            </button>
          </div>

          {/* Caption bar */}
          <div className="flex flex-wrap items-center gap-3 px-3 py-2 bg-slate-800/90 rounded-xl text-sm">
            <span className="font-semibold text-white">{species}</span>
            <span className="text-emerald-400 font-mono">{confidencePct}%</span>
            <span className="text-slate-400">{timeAgo(timestamp)}</span>
            <a
              href={fullUrl}
              download
              className="ml-auto rounded-md bg-slate-700 px-3 py-1.5 text-xs font-medium text-slate-100 transition-colors hover:bg-slate-600"
              onClick={(e) => e.stopPropagation()}
            >
              Download
            </a>
            <button
              className="rounded-md bg-red-600/90 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-red-500 disabled:opacity-50"
              onClick={(e) => { e.stopPropagation(); handleDelete(); }}
              disabled={deleting}
            >
              {deleting ? "Deleting…" : "Delete"}
            </button>
            {deleteError && (
              <span className="w-full text-xs text-red-400">{deleteError}</span>
            )}
          </div>
        </div>

        {/* ── Reference panel — locked to the image's exact rendered size ── */}
        {showReference && imgSize && (
          <div
            className="shrink-0 overflow-y-auto rounded-lg bg-slate-900 shadow-2xl p-4"
            style={{ width: imgSize.w, height: imgSize.h }}
          >
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-3">
              Reference
            </h3>
            <ReferencePane
              state={refState}
              activeImageIndex={activeImageIndex}
              onSelectImage={setActiveImageIndex}
            />
          </div>
        )}
      </div>

      {/* Next arrow */}
      {onNext && (
        <button
          className="absolute right-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-slate-800/80 hover:bg-slate-700 text-white text-2xl transition-colors z-10"
          onClick={(e) => { e.stopPropagation(); onNext(); }}
          aria-label="Next detection"
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
      <div className="flex flex-col items-center justify-center min-h-[200px] gap-3 text-slate-500">
        <div
          className="w-8 h-8 border-2 border-slate-600 border-t-emerald-500 rounded-full animate-spin"
          aria-hidden="true"
        />
        <p className="text-sm animate-pulse">Loading reference…</p>
      </div>
    );
  }

  if (state.kind === "none") {
    return (
      <div className="flex items-center justify-center min-h-[200px] text-center px-4">
        <p className="text-sm text-slate-500">
          No reference available for this species yet.
        </p>
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="flex items-center justify-center min-h-[200px] text-center px-4">
        <p className="text-sm text-amber-400">
          Couldn’t load reference: {state.message}
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
            className="w-full object-contain rounded-lg bg-black"
          />
          <p className="text-[11px] text-slate-500 leading-snug">
            {activeImage.attribution}
            {activeImage.license ? ` · ${activeImage.license}` : ""}
          </p>
        </div>
      ) : (
        <div className="flex items-center justify-center h-32 rounded-lg bg-slate-800 text-sm text-slate-500">
          No reference image
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
                  ? "border-emerald-500"
                  : "border-transparent hover:border-slate-500"
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
          <p className="text-base font-semibold text-white">{reference.common_name}</p>
          {reference.scientific_name && (
            <p className="text-sm italic text-slate-400">{reference.scientific_name}</p>
          )}
        </div>

        {reference.summary && (
          <p className="text-sm text-slate-300 leading-relaxed">{reference.summary}</p>
        )}

        {reference.behaviour && (
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-1">
              Behaviour
            </p>
            <p className="text-sm text-slate-300 leading-relaxed">{reference.behaviour}</p>
          </div>
        )}

        {reference.wikipedia_url && (
          <a
            href={reference.wikipedia_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-emerald-400 hover:text-emerald-300 underline w-fit"
          >
            Read on Wikipedia →
          </a>
        )}
      </div>
    </div>
  );
}
