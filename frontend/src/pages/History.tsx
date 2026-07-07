import { useEffect, useState, useCallback } from "react";
import { api, type Detection, type SpeciesSummary } from "../api";
import { Timeline } from "../components/Timeline";
import { Gallery } from "../components/Gallery";

const PAGE_SIZE = 20;

type Tab = "timeline" | "gallery";

/**
 * Full-page history view with:
 * - Filter bar: species dropdown populated from /api/species + date range pickers
 * - Tabs: Timeline (chronological list) vs Gallery (thumbnail grid)
 * - Infinite scroll within each sub-view (20 detections per page)
 * - Lightbox for full-res images (never auto-loaded)
 * - Bulk-select + ZIP download (Gallery only)
 */
export function History() {
  // ── Filter state ────────────────────────────────────────────────
  const [speciesList, setSpeciesList] = useState<SpeciesSummary[]>([]);
  const [filterSpecies, setFilterSpecies] = useState<string>("");
  const [filterFrom, setFilterFrom] = useState<string>("");
  const [filterTo, setFilterTo] = useState<string>("");
  // Minimum confidence as a 0–100 percentage; 0 means "show all".
  // `sliderMinConfidence` tracks the live slider position for display only;
  // `filterMinConfidence` is the committed value that drives the fetch and is
  // only updated when the slider is released (so dragging doesn't refetch on
  // every intermediate value).
  const [sliderMinConfidence, setSliderMinConfidence] = useState<number>(0);
  const [filterMinConfidence, setFilterMinConfidence] = useState<number>(0);

  // ── Pagination state ─────────────────────────────────────────────
  const [detections, setDetections] = useState<Detection[]>([]);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [exhausted, setExhausted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── UI state ─────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<Tab>("timeline");
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  // ── Load species dropdown once ────────────────────────────────────
  useEffect(() => {
    api.species.list().then(setSpeciesList).catch(() => {/* non-critical */});
  }, []);

  // ── Initial/filter-change load ────────────────────────────────────
  /** Fetch the first page of results for the current filters. */
  const loadFirstPage = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDetections([]);
    setOffset(0);
    setExhausted(false);
    setSelectedIds(new Set());
    setLightboxIndex(null);

    try {
      const data = await api.detections.list({
        species: filterSpecies || undefined,
        from: filterFrom || undefined,
        to: filterTo || undefined,
        min_confidence: filterMinConfidence > 0 ? filterMinConfidence / 100 : undefined,
        limit: PAGE_SIZE,
        offset: 0,
      });

      setDetections(data);
      setOffset(data.length);
      if (data.length < PAGE_SIZE) setExhausted(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load detections");
    } finally {
      setLoading(false);
    }
  }, [filterSpecies, filterFrom, filterTo, filterMinConfidence]);

  useEffect(() => {
    loadFirstPage();
  }, [loadFirstPage]);

  // ── Infinite scroll — load next page ─────────────────────────────
  const handleLoadMore = useCallback(async () => {
    if (loadingMore || exhausted) return;

    setLoadingMore(true);
    try {
      const data = await api.detections.list({
        species: filterSpecies || undefined,
        from: filterFrom || undefined,
        to: filterTo || undefined,
        min_confidence: filterMinConfidence > 0 ? filterMinConfidence / 100 : undefined,
        limit: PAGE_SIZE,
        offset,
      });

      // Drop any rows we already hold before appending. The detector writes
      // new detections live, so a row inserted between page fetches shifts the
      // offset window and makes the next page re-return rows we already have.
      // Without this guard those rows render twice (in both Timeline and
      // Gallery, which share this list). The offset cursor still advances by
      // the raw page length so the server-side window keeps moving forward.
      setDetections((prev) => {
        const seen = new Set(prev.map((d) => d.id));
        const fresh = data.filter((d) => !seen.has(d.id));
        return [...prev, ...fresh];
      });
      setOffset((prev) => prev + data.length);
      if (data.length < PAGE_SIZE) setExhausted(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load more detections");
    } finally {
      setLoadingMore(false);
    }
  }, [loadingMore, exhausted, filterSpecies, filterFrom, filterTo, filterMinConfidence, offset]);

  // ── Lightbox helpers ──────────────────────────────────────────────
  function handleOpenLightbox(index: number) {
    setLightboxIndex(index);
  }

  function handleCloseLightbox() {
    setLightboxIndex(null);
  }

  // ── Deletion ──────────────────────────────────────────────────────
  /**
   * Remove already-deleted detections from local state. The actual API
   * deletes are performed by the Lightbox / FileDownloader; this only reaps
   * the now-gone rows so the list, selection, and pagination offset stay in
   * sync, and closes the lightbox (its positional index is no longer valid).
   */
  const removeDetections = useCallback((ids: number[]) => {
    if (ids.length === 0) return;
    const removed = new Set(ids);
    setDetections((prev) => prev.filter((d) => !removed.has(d.id)));
    setSelectedIds((prev) => {
      const next = new Set(prev);
      ids.forEach((id) => next.delete(id));
      return next;
    });
    setOffset((prev) => Math.max(0, prev - ids.length));
    setLightboxIndex(null);
  }, []);

  /** Remove a single deleted detection (used by the Lightbox). */
  const handleDeleteDetection = useCallback(
    (id: number) => removeDetections([id]),
    [removeDetections]
  );

  /**
   * Replace a corrected detection in place so its new species shows in the
   * Timeline/Gallery immediately (the API call happens in the Lightbox). Unlike
   * a delete, this leaves the offset, selection, and lightbox index untouched.
   */
  const handleUpdateDetection = useCallback((updated: Detection) => {
    setDetections((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
  }, []);

  const fieldLabel = "text-xs font-semibold text-sage-deep";
  const fieldInput =
    "bg-paper text-ink text-sm rounded-lg px-3 py-2 border border-line focus:outline-none focus:ring-2 focus:ring-gold focus:border-gold";

  return (
    <div className="mx-auto max-w-6xl">
      {/* Page header */}
      <div className="px-6 pt-8 pb-4 space-y-5">
        <header>
          <p className="eyebrow mb-2">The log</p>
          <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">
            Sightings log
          </h1>
          <p className="mt-1 text-sm text-bark">Every bird the feeder has noted down.</p>
        </header>

        {/* ── Filter bar ─────────────────────────────────────────── */}
        <div className="flex flex-wrap items-end gap-3 rounded-2xl border border-line bg-card p-4 shadow-plate">
          {/* Species dropdown */}
          <div className="flex flex-col gap-1 min-w-[160px]">
            <label className={fieldLabel} htmlFor="filter-species">
              Species
            </label>
            <select
              id="filter-species"
              className={fieldInput}
              value={filterSpecies}
              onChange={(e) => setFilterSpecies(e.target.value)}
            >
              <option value="">All species</option>
              {speciesList.map((s) => (
                <option key={s.species} value={s.species}>
                  {s.species} ({s.count})
                </option>
              ))}
            </select>
          </div>

          {/* Date from */}
          <div className="flex flex-col gap-1">
            <label className={fieldLabel} htmlFor="filter-from">
              From
            </label>
            <input
              id="filter-from"
              type="date"
              className={fieldInput}
              value={filterFrom}
              onChange={(e) => setFilterFrom(e.target.value)}
            />
          </div>

          {/* Date to */}
          <div className="flex flex-col gap-1">
            <label className={fieldLabel} htmlFor="filter-to">
              To
            </label>
            <input
              id="filter-to"
              type="date"
              className={fieldInput}
              value={filterTo}
              onChange={(e) => setFilterTo(e.target.value)}
            />
          </div>

          {/* Minimum confidence slider — refetches only when released. */}
          <div className="flex flex-col gap-1 min-w-[180px]">
            <label className={fieldLabel} htmlFor="filter-confidence">
              Match above <span className="tnum text-gold-deep">{sliderMinConfidence}%</span>
            </label>
            <input
              id="filter-confidence"
              type="range"
              min={0}
              max={100}
              step={1}
              className="mt-2 accent-gold"
              value={sliderMinConfidence}
              onChange={(e) => setSliderMinConfidence(Number(e.target.value))}
              onMouseUp={() => setFilterMinConfidence(sliderMinConfidence)}
              onTouchEnd={() => setFilterMinConfidence(sliderMinConfidence)}
              onKeyUp={() => setFilterMinConfidence(sliderMinConfidence)}
            />
          </div>

          {/* Clear filters */}
          {(filterSpecies || filterFrom || filterTo || filterMinConfidence > 0) && (
            <button
              className="self-end pb-2 text-sm text-bark underline hover:text-ink"
              onClick={() => {
                setFilterSpecies("");
                setFilterFrom("");
                setFilterTo("");
                setSliderMinConfidence(0);
                setFilterMinConfidence(0);
              }}
            >
              Clear filters
            </button>
          )}
        </div>

        {/* ── Tab switcher ─────────────────────────────────────────── */}
        <div className="flex w-fit gap-1 rounded-xl border border-line bg-card p-1">
          {(["timeline", "gallery"] as Tab[]).map((tab) => (
            <button
              key={tab}
              className={`rounded-lg px-4 py-1.5 text-sm font-medium capitalize transition-colors ${
                activeTab === tab
                  ? "bg-gold text-card shadow-sm"
                  : "text-bark hover:text-ink"
              }`}
              onClick={() => setActiveTab(tab)}
            >
              {tab === "timeline" ? "List" : "Grid"}
            </button>
          ))}
        </div>
      </div>

      {/* ── Sub-view ──────────────────────────────────────────────────── */}
      <div className="px-6 pb-12">
        {activeTab === "timeline" ? (
          <Timeline
            detections={detections}
            loading={loading}
            loadingMore={loadingMore}
            exhausted={exhausted}
            error={error}
            onLoadMore={handleLoadMore}
            lightboxIndex={lightboxIndex}
            onOpenLightbox={handleOpenLightbox}
            onCloseLightbox={handleCloseLightbox}
            onDeleteDetection={handleDeleteDetection}
            onUpdateDetection={handleUpdateDetection}
          />
        ) : (
          <Gallery
            detections={detections}
            loading={loading}
            loadingMore={loadingMore}
            exhausted={exhausted}
            error={error}
            onLoadMore={handleLoadMore}
            selectedIds={selectedIds}
            onSelectionChange={setSelectedIds}
            lightboxIndex={lightboxIndex}
            onOpenLightbox={handleOpenLightbox}
            onCloseLightbox={handleCloseLightbox}
            onDeleteDetection={handleDeleteDetection}
            onUpdateDetection={handleUpdateDetection}
            onDeleteSelected={removeDetections}
          />
        )}
      </div>
    </div>
  );
}
