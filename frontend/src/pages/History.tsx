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
  }, [filterSpecies, filterFrom, filterTo]);

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
        limit: PAGE_SIZE,
        offset,
      });

      setDetections((prev) => [...prev, ...data]);
      setOffset((prev) => prev + data.length);
      if (data.length < PAGE_SIZE) setExhausted(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load more detections");
    } finally {
      setLoadingMore(false);
    }
  }, [loadingMore, exhausted, filterSpecies, filterFrom, filterTo, offset]);

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

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Page header */}
      <div className="px-6 pt-6 pb-4 space-y-4">
        <h1 className="text-2xl font-bold tracking-tight">Detection History</h1>

        {/* ── Filter bar ─────────────────────────────────────────── */}
        <div className="flex flex-wrap items-end gap-3 bg-slate-800 rounded-2xl p-4">
          {/* Species dropdown */}
          <div className="flex flex-col gap-1 min-w-[160px]">
            <label className="text-xs text-slate-400 font-medium" htmlFor="filter-species">
              Species
            </label>
            <select
              id="filter-species"
              className="bg-slate-700 text-white text-sm rounded-lg px-3 py-2 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
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
            <label className="text-xs text-slate-400 font-medium" htmlFor="filter-from">
              From
            </label>
            <input
              id="filter-from"
              type="date"
              className="bg-slate-700 text-white text-sm rounded-lg px-3 py-2 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              value={filterFrom}
              onChange={(e) => setFilterFrom(e.target.value)}
            />
          </div>

          {/* Date to */}
          <div className="flex flex-col gap-1">
            <label className="text-xs text-slate-400 font-medium" htmlFor="filter-to">
              To
            </label>
            <input
              id="filter-to"
              type="date"
              className="bg-slate-700 text-white text-sm rounded-lg px-3 py-2 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-emerald-500"
              value={filterTo}
              onChange={(e) => setFilterTo(e.target.value)}
            />
          </div>

          {/* Clear filters */}
          {(filterSpecies || filterFrom || filterTo) && (
            <button
              className="text-sm text-slate-400 hover:text-white underline self-end pb-2"
              onClick={() => {
                setFilterSpecies("");
                setFilterFrom("");
                setFilterTo("");
              }}
            >
              Clear filters
            </button>
          )}
        </div>

        {/* ── Tab switcher ─────────────────────────────────────────── */}
        <div className="flex gap-1 bg-slate-800 rounded-xl p-1 w-fit">
          {(["timeline", "gallery"] as Tab[]).map((tab) => (
            <button
              key={tab}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium capitalize transition-colors ${
                activeTab === tab
                  ? "bg-emerald-600 text-white"
                  : "text-slate-400 hover:text-white"
              }`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* ── Sub-view ──────────────────────────────────────────────────── */}
      <div className="px-6 pb-10">
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
            onDeleteSelected={removeDetections}
          />
        )}
      </div>
    </div>
  );
}
