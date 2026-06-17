import { useEffect, useState } from "react";
import { api, type Detection } from "../api";
import { DetectionCard } from "../components/DetectionCard";
import { SystemMonitor } from "../components/SystemMonitor";
import { Lightbox } from "../components/Lightbox";

// "Recent Predictions" shows this many of the most recent detections from
// before today. "Predictions Today" is capped higher so a busy day still
// renders without an unbounded query.
const RECENT_LIMIT = 16;
const TODAY_LIMIT = 100;

/** Which dashboard strip a lightbox selection belongs to. */
type Section = "today" | "recent";

/** Format a Date as a timezone-less ("naive") local ISO string.
 *
 * Detection timestamps are written with `datetime.now()` (naive local time)
 * and the API compares the `from`/`to` query params against them directly, so
 * the bounds must also be naive local time — emitting a `Z`/offset here would
 * make pydantic parse them as UTC and shift the day boundary.
 */
function toNaiveISO(d: Date): string {
  const pad = (n: number): string => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}

export function Dashboard() {
  const [today, setToday] = useState<Detection[]>([]);
  const [recent, setRecent] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Minimum confidence as a 0–100 percentage; 0 means "show all".
  // `sliderConfidence` tracks the live slider position for display only;
  // `minConfidence` is the committed value that drives the fetch and is only
  // updated when the slider is released (so dragging doesn't refetch on every
  // intermediate value).
  const [sliderConfidence, setSliderConfidence] = useState<number>(0);
  const [minConfidence, setMinConfidence] = useState<number>(0);
  // Which strip + index is open in the comparison panel, or null when closed.
  const [lightbox, setLightbox] = useState<{ section: Section; index: number } | null>(null);

  useEffect(() => {
    setLoading(true);
    // Local midnight is the boundary between "today" and "recent". The two
    // queries are disjoint: today is everything at/after midnight, recent is
    // the latest rows strictly before it (one second before midnight as an
    // inclusive `to` bound).
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const beforeToday = new Date(todayStart.getTime() - 1000);
    const minConf = minConfidence > 0 ? minConfidence / 100 : undefined;

    Promise.all([
      api.detections.list({
        from: toNaiveISO(todayStart),
        limit: TODAY_LIMIT,
        min_confidence: minConf,
      }),
      api.detections.list({
        to: toNaiveISO(beforeToday),
        limit: RECENT_LIMIT,
        min_confidence: minConf,
      }),
    ])
      .then(([todayData, recentData]) => {
        setToday(todayData);
        setRecent(recentData);
        setError(null);
        setLoading(false);
        setLightbox(null);
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : "Failed to load detections");
        setLoading(false);
      });
  }, [minConfidence]);

  const activeList = lightbox?.section === "today" ? today : recent;
  const currentDetection = lightbox !== null ? activeList[lightbox.index] ?? null : null;

  const removeDetection = (deletedId: number): void => {
    setToday((prev) => prev.filter((d) => d.id !== deletedId));
    setRecent((prev) => prev.filter((d) => d.id !== deletedId));
    setLightbox(null);
  };

  /** Render one horizontal strip of detection cards (or its empty/loading state). */
  const renderStrip = (section: Section, list: Detection[], emptyLabel: string) => {
    if (loading) {
      return <p className="text-sm text-slate-500 animate-pulse">Loading…</p>;
    }
    if (list.length === 0) {
      return <p className="text-sm text-slate-500">{emptyLabel}</p>;
    }
    return (
      <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-700">
        {list.map((d, i) => (
          <DetectionCard
            key={d.id}
            detection={d}
            onOpenLightbox={() => setLightbox({ section, index: i })}
          />
        ))}
      </div>
    );
  };

  const noConfidenceSuffix =
    minConfidence > 0 ? ` at or above ${minConfidence}% confidence` : "";

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6 space-y-6">
      {/* Comparison panel overlay */}
      {currentDetection && lightbox !== null && (
        <Lightbox
          detection={currentDetection}
          onClose={() => setLightbox(null)}
          onPrev={
            lightbox.index > 0
              ? () => setLightbox({ section: lightbox.section, index: lightbox.index - 1 })
              : null
          }
          onNext={
            lightbox.index < activeList.length - 1
              ? () => setLightbox({ section: lightbox.section, index: lightbox.index + 1 })
              : null
          }
          onDelete={removeDetection}
        />
      )}

      <header>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
      </header>

      <SystemMonitor />

      {/* Minimum confidence slider applies to both strips below. */}
      <div className="flex items-center justify-end gap-3">
        <label
          className="text-xs text-slate-400 font-medium whitespace-nowrap"
          htmlFor="dashboard-confidence"
        >
          Min confidence: {sliderConfidence}%
        </label>
        <input
          id="dashboard-confidence"
          type="range"
          min={0}
          max={100}
          step={1}
          className="accent-emerald-500 w-40"
          value={sliderConfidence}
          onChange={(e) => setSliderConfidence(Number(e.target.value))}
          onMouseUp={() => setMinConfidence(sliderConfidence)}
          onTouchEnd={() => setMinConfidence(sliderConfidence)}
          onKeyUp={() => setMinConfidence(sliderConfidence)}
        />
      </div>

      {error && <p className="text-sm text-red-400">{error}</p>}

      <section>
        <h2 className="text-lg font-semibold mb-3">Predictions Today</h2>
        {renderStrip("today", today, `No detections today${noConfidenceSuffix}.`)}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Recent Predictions</h2>
        {renderStrip("recent", recent, `No earlier detections${noConfidenceSuffix}.`)}
      </section>
    </div>
  );
}
