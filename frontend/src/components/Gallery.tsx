import { type Detection } from "../api";
import { DetectionCard } from "./DetectionCard";
import { Lightbox } from "./Lightbox";
import { FileDownloader } from "./FileDownloader";
import { useRef, useEffect, useCallback } from "react";

interface GalleryProps {
  /** Detections loaded so far (all pages combined). */
  detections: Detection[];
  /** Whether the first page is still loading. */
  loading: boolean;
  /** Whether a subsequent page is being fetched. */
  loadingMore: boolean;
  /** Whether all pages have been fetched. */
  exhausted: boolean;
  /** Error message, if any. */
  error: string | null;
  /** Called to trigger the next page. */
  onLoadMore: () => void;
  /** Currently selected detection IDs for bulk download. */
  selectedIds: Set<number>;
  /** Called when selection changes. */
  onSelectionChange: (ids: Set<number>) => void;
  /** Index of the detection currently open in the lightbox, or null if closed. */
  lightboxIndex: number | null;
  /** Called when a card is clicked to open the lightbox. */
  onOpenLightbox: (index: number) => void;
  /** Called when the lightbox is closed. */
  onCloseLightbox: () => void;
  /** Called with a detection id after it has been deleted from the lightbox. */
  onDeleteDetection: (id: number) => void;
  /** Called with the deleted ids after a bulk delete of the selection. */
  onDeleteSelected: (ids: number[]) => void;
}

/**
 * Uniform thumbnail grid with checkbox-based bulk selection and infinite scroll.
 *
 * Each card can be clicked to toggle its selection (shown via a checkbox
 * overlay + ring). A FileDownloader action bar is rendered above the grid
 * when items are selected or as a persistent toolbar for "Select all".
 * Full-res images are never loaded here — only in the Lightbox.
 */
export function Gallery({
  detections,
  loading,
  loadingMore,
  exhausted,
  error,
  onLoadMore,
  selectedIds,
  onSelectionChange,
  lightboxIndex,
  onOpenLightbox,
  onCloseLightbox,
  onDeleteDetection,
  onDeleteSelected,
}: GalleryProps) {
  const sentinelRef = useRef<HTMLDivElement>(null);

  /** Toggle a single detection's selection state. */
  function handleCardSelect(detection: Detection) {
    const next = new Set(selectedIds);
    if (next.has(detection.id)) {
      next.delete(detection.id);
    } else {
      next.add(detection.id);
    }
    onSelectionChange(next);
  }

  // IntersectionObserver for infinite scroll
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
        />
      )}

      <div className="flex flex-col gap-4">
        {/* Bulk-download toolbar */}
        <div className="bg-slate-800/60 rounded-xl p-3">
          <FileDownloader
            allDetections={detections}
            selectedIds={selectedIds}
            onSelectionChange={onSelectionChange}
            onDeleteSelected={onDeleteSelected}
          />
        </div>

        {loading && (
          <p className="text-sm text-slate-500 animate-pulse">Loading…</p>
        )}

        {error && (
          <p className="text-sm text-red-400">{error}</p>
        )}

        {!loading && !error && detections.length === 0 && (
          <p className="text-sm text-slate-500">No detections match your filters.</p>
        )}

        {/* Thumbnail grid */}
        <div className="flex flex-wrap gap-3">
          {detections.map((d, i) => (
            <DetectionCard
              key={d.id}
              detection={d}
              selected={selectedIds.has(d.id)}
              onSelect={handleCardSelect}
              onOpenLightbox={() => onOpenLightbox(i)}
            />
          ))}
        </div>

        {loadingMore && (
          <p className="text-sm text-slate-500 animate-pulse text-center py-4">
            Loading more…
          </p>
        )}

        {!loading && !loadingMore && exhausted && detections.length > 0 && (
          <p className="text-sm text-slate-600 text-center py-4">
            All {detections.length} detections loaded.
          </p>
        )}

        {/* Scroll sentinel */}
        <div ref={sentinelRef} className="h-1" aria-hidden="true" />
      </div>
    </>
  );
}
