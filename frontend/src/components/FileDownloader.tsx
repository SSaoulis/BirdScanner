import { useState } from "react";
import { api, type Detection } from "../api";

interface FileDownloaderProps {
  /** All detections currently filtered/visible — used for "Select all" shortcut. */
  allDetections: Detection[];
  /** Currently selected detection IDs. */
  selectedIds: Set<number>;
  /** Called when the selection changes. */
  onSelectionChange: (ids: Set<number>) => void;
  /** Called with the ids that were successfully deleted after a bulk delete. */
  onDeleteSelected: (ids: number[]) => void;
}

interface DownloadProgress {
  loaded: number;
  total: number | null;
  done: boolean;
}

/**
 * Manages bulk-select state and ZIP download with progress tracking.
 *
 * Renders a sticky action bar that appears when one or more detections
 * are selected. Uses `fetch()` with a `ReadableStream` to track progress
 * via the `Content-Length` header.
 */
export function FileDownloader({ allDetections, selectedIds, onSelectionChange, onDeleteSelected }: FileDownloaderProps) {
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const selectedCount = selectedIds.size;
  const allIds = allDetections.map((d) => d.id);
  const allSelected = allIds.length > 0 && allIds.every((id) => selectedIds.has(id));

  /** Toggle selection of every visible detection. */
  function handleSelectAll() {
    if (allSelected) {
      onSelectionChange(new Set());
    } else {
      onSelectionChange(new Set(allIds));
    }
  }

  /** Clear the current selection. */
  function handleClearSelection() {
    onSelectionChange(new Set());
  }

  /**
   * Stream the ZIP from the server and track byte progress using
   * the `Content-Length` response header.
   */
  async function handleDownload() {
    if (selectedIds.size === 0) return;

    setError(null);
    setProgress({ loaded: 0, total: null, done: false });

    try {
      const url = api.images.downloadUrl(Array.from(selectedIds));
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Download failed: ${response.status} ${response.statusText}`);
      }

      const contentLength = response.headers.get("Content-Length");
      const total = contentLength ? parseInt(contentLength, 10) : null;

      if (!response.body) {
        throw new Error("Response body is not readable");
      }

      const reader = response.body.getReader();
      const chunks: Uint8Array<ArrayBuffer>[] = [];
      let loaded = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        setProgress({ loaded, total, done: false });
      }

      // Assemble full buffer and trigger browser download
      const blob = new Blob(chunks, { type: "application/zip" });
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = `bird_detections_${Date.now()}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(objectUrl);

      setProgress({ loaded, total, done: true });
      setTimeout(() => setProgress(null), 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Download failed");
      setProgress(null);
    }
  }

  /**
   * Confirm, then delete every selected detection. Deletes are issued
   * sequentially; ids that succeed are reported to the parent even if a later
   * one fails, so a partial failure still removes what it could.
   */
  async function handleDeleteSelected() {
    if (selectedIds.size === 0 || deleting) return;
    if (!window.confirm(`Permanently delete ${selectedCount} detection(s) and their images?`)) {
      return;
    }

    setDeleting(true);
    setError(null);
    const deleted: number[] = [];
    let failed = 0;
    for (const id of Array.from(selectedIds)) {
      try {
        await api.detections.delete(id);
        deleted.push(id);
      } catch {
        failed += 1;
      }
    }
    if (deleted.length > 0) onDeleteSelected(deleted);
    if (failed > 0) setError(`Failed to delete ${failed} detection(s).`);
    setDeleting(false);
  }

  /** Compute a 0–100 integer percentage or null if total is unknown. */
  function getPercent(p: DownloadProgress): number | null {
    if (p.total === null) return null;
    return Math.min(Math.round((p.loaded / p.total) * 100), 100);
  }

  return (
    <div className="flex flex-wrap items-center gap-3">
      {/* Select / deselect all */}
      <button
        className="text-sm px-3 py-1.5 rounded-lg bg-slate-700 hover:bg-slate-600 text-white transition-colors"
        onClick={handleSelectAll}
        disabled={allIds.length === 0}
      >
        {allSelected ? "Deselect all" : `Select all (${allIds.length})`}
      </button>

      {/* Selection status + clear */}
      {selectedCount > 0 && (
        <>
          <span className="text-sm text-slate-300">
            {selectedCount} selected
          </span>
          <button
            className="text-sm text-slate-400 hover:text-white underline"
            onClick={handleClearSelection}
          >
            Clear
          </button>
        </>
      )}

      {/* Download button */}
      {selectedCount > 0 && (
        <button
          className="ml-auto flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-semibold text-sm transition-colors"
          onClick={handleDownload}
          disabled={progress !== null && !progress.done}
        >
          {progress && !progress.done ? "Downloading…" : `Download ZIP (${selectedCount})`}
        </button>
      )}

      {/* Delete button */}
      {selectedCount > 0 && (
        <button
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white font-semibold text-sm transition-colors"
          onClick={handleDeleteSelected}
          disabled={deleting}
        >
          {deleting ? "Deleting…" : `Delete (${selectedCount})`}
        </button>
      )}

      {/* Progress bar */}
      {progress && !progress.done && (
        <div className="w-full mt-1">
          <div className="h-2 rounded-full bg-slate-700 overflow-hidden">
            {getPercent(progress) !== null ? (
              <div
                className="h-full rounded-full bg-emerald-500 transition-all duration-200"
                style={{ width: `${getPercent(progress)}%` }}
              />
            ) : (
              /* Indeterminate bar when Content-Length is unknown */
              <div className="h-full w-1/3 rounded-full bg-emerald-500 animate-pulse" />
            )}
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            {getPercent(progress) !== null
              ? `${getPercent(progress)}%`
              : `${(progress.loaded / 1024).toFixed(0)} KB downloaded…`}
          </p>
        </div>
      )}

      {progress?.done && (
        <p className="w-full text-xs text-emerald-400 mt-0.5">Download complete.</p>
      )}

      {error && (
        <p className="w-full text-xs text-red-400 mt-0.5">{error}</p>
      )}
    </div>
  );
}
