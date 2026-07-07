import { useEffect, useState } from "react";
import { api, type Detection } from "../api";
import { DetectionCard } from "../components/DetectionCard";
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

  // Replace the corrected detection in place in both strips, keeping the
  // lightbox open on it so its species/reference refresh live.
  const updateDetection = (updated: Detection): void => {
    setToday((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
    setRecent((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
  };

  /** Render one horizontal strip of detection cards (or its empty/loading state). */
  const renderStrip = (section: Section, list: Detection[], emptyLabel: string) => {
    if (loading) {
      return <p className="text-sm text-bark animate-pulse">Checking the feeder…</p>;
    }
    if (list.length === 0) {
      return <p className="text-sm text-bark">{emptyLabel}</p>;
    }
    return (
      <div className="flex gap-4 overflow-x-auto pb-3">
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
    minConfidence > 0 ? ` at ${minConfidence}% match or better` : "";

  const todayDate = new Date();
  const todayLabel = todayDate.toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
  });

  return (
    <div className="mx-auto max-w-6xl px-6 py-8 space-y-8">
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
          onUpdate={updateDetection}
        />
      )}

      <header>
        <p className="eyebrow mb-2">At the feeder</p>
        <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">
          Today&rsquo;s visitors
        </h1>
        <p className="mt-1 text-sm text-bark">{todayLabel}</p>
      </header>

      {/* Minimum confidence slider applies to both strips below. */}
      <div className="flex items-center justify-end gap-3">
        <label
          className="text-xs font-semibold text-sage-deep whitespace-nowrap"
          htmlFor="dashboard-confidence"
        >
          Only show matches above{" "}
          <span className="tnum text-gold-deep">{sliderConfidence}%</span>
        </label>
        <input
          id="dashboard-confidence"
          type="range"
          min={0}
          max={100}
          step={1}
          className="w-40 accent-gold"
          value={sliderConfidence}
          onChange={(e) => setSliderConfidence(Number(e.target.value))}
          onMouseUp={() => setMinConfidence(sliderConfidence)}
          onTouchEnd={() => setMinConfidence(sliderConfidence)}
          onKeyUp={() => setMinConfidence(sliderConfidence)}
        />
      </div>

      {error && <p className="text-sm text-rust">{error}</p>}

      <section className="space-y-4">
        <h2 className="eyebrow">Spotted today</h2>
        {renderStrip("today", today, `No birds spotted today${noConfidenceSuffix}.`)}
      </section>

      <section className="space-y-4">
        <h2 className="eyebrow">Earlier sightings</h2>
        {renderStrip("recent", recent, `Nothing earlier to show${noConfidenceSuffix}.`)}
      </section>
    </div>
  );
}
