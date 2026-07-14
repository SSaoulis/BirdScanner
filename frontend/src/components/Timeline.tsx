import { useEffect, useRef, useCallback } from "react";
import { type Detection } from "../api";
import { DetectionCard } from "./DetectionCard";
import { Lightbox } from "./Lightbox";

interface TimelineProps {
  /** Detections loaded so far (all pages combined). */
  detections: Detection[];
  /** Whether the very first page is still loading. */
  loading: boolean;
  /** Whether an additional page is being fetched. */
  loadingMore: boolean;
  /** Whether all pages have been fetched (no more to load). */
  exhausted: boolean;
  /** Error message, if any. */
  error: string | null;
  /** Called when the sentinel element scrolls into view — triggers next page fetch. */
  onLoadMore: () => void;
  /** Index of the detection currently open in the lightbox, or null if closed. */
  lightboxIndex: number | null;
  /** Called when a card thumbnail is clicked to open the lightbox. */
  onOpenLightbox: (index: number) => void;
  /** Called when the lightbox is closed. */
  onCloseLightbox: () => void;
  /** Called with a detection id after it has been deleted from the lightbox. */
  onDeleteDetection: (id: number) => void;
  /** Called with the updated detection after its species is corrected. */
  onUpdateDetection: (updated: Detection) => void;
}

/**
 * Chronological paginated list of DetectionCards with infinite scroll.
 *
 * Uses an `IntersectionObserver` on a sentinel `<div>` at the bottom of the
 * list to trigger `onLoadMore` when the user scrolls to the end. Full-res
 * images are only loaded when the lightbox is explicitly opened.
 */
export function Timeline({
  detections,
  loading,
  loadingMore,
  exhausted,
  error,
  onLoadMore,
  lightboxIndex,
  onOpenLightbox,
  onCloseLightbox,
  onDeleteDetection,
  onUpdateDetection,
}: TimelineProps) {
  const sentinelRef = useRef<HTMLDivElement>(null);

  // IntersectionObserver to detect when user has scrolled to the bottom
  const observerCallback = useCallback(
    (entries: IntersectionObserverEntry[]) => {
      if (entries[0].isIntersecting && !loadingMore && !exhausted) {
        onLoadMore();
      }
    },
    [loadingMore, exhausted, onLoadMore]
  );

  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;

    const observer = new IntersectionObserver(observerCallback, { threshold: 0.1 });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [observerCallback]);

  const currentDetection = lightboxIndex !== null ? detections[lightboxIndex] ?? null : null;

  return (
    <>
      {/* Lightbox overlay */}
      {currentDetection && (
        <Lightbox
          detection={currentDetection}
          onClose={onCloseLightbox}
          onPrev={lightboxIndex !== null && lightboxIndex > 0 ? () => onOpenLightbox(lightboxIndex! - 1) : null}
          onNext={
            lightboxIndex !== null && lightboxIndex < detections.length - 1
              ? () => onOpenLightbox(lightboxIndex! + 1)
              : null
          }
          onDelete={onDeleteDetection}
          onUpdate={onUpdateDetection}
          position={
            lightboxIndex !== null
              ? { index: lightboxIndex, total: detections.length }
              : null
          }
        />
      )}

      <div className="flex flex-col gap-4">
        {loading && (
          <p className="text-sm text-bark animate-pulse">Leafing through the log…</p>
        )}

        {error && (
          <p className="text-sm text-rust">{error}</p>
        )}

        {!loading && !error && detections.length === 0 && (
          <p className="text-sm text-bark">No sightings match these filters.</p>
        )}

        {/* Card grid — full-width rows on mobile, uniform plates on `sm`+ */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-4">
          {detections.map((d, i) => (
            <DetectionCard
              key={d.id}
              detection={d}
              onOpenLightbox={() => onOpenLightbox(i)}
            />
          ))}
        </div>

        {/* Load-more spinner */}
        {loadingMore && (
          <p className="text-sm text-bark animate-pulse text-center py-4">
            Loading more…
          </p>
        )}

        {/* Exhausted message */}
        {!loading && !loadingMore && exhausted && detections.length > 0 && (
          <p className="text-sm text-sage-deep text-center py-4">
            That&rsquo;s all {detections.length} sightings.
          </p>
        )}

        {/* Sentinel element watched by IntersectionObserver */}
        <div ref={sentinelRef} className="h-1" aria-hidden="true" />
      </div>
    </>
  );
}
