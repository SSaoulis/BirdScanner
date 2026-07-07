import { useEffect, useState } from "react";
import { api, type Detection } from "../api";
import { DetectionCard } from "../components/DetectionCard";
import { Lightbox } from "../components/Lightbox";

// "Spotted today" is capped higher so a busy day still renders without an
// unbounded query. The earlier sightings (this week / this month / older) share
// a single fetch of detections from before today, capped here — it is larger
// than the old single-row limit because the list is now split into a grid of
// time-bucketed groups rather than one horizontal strip.
const TODAY_LIMIT = 100;
const EARLIER_LIMIT = 120;

/** Which dashboard strip a lightbox selection belongs to. */
type Section = "today" | "earlier";

/** A time bucket of earlier sightings plus the cards it holds. */
interface EarlierGroup {
  /** Section heading, e.g. "This week". */
  label: string;
  /**
   * Each card paired with its position in the flat `earlier` list, so the
   * lightbox's prev/next (which index into that list) traverse seamlessly
   * across group boundaries.
   */
  items: { detection: Detection; index: number }[];
}

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

/**
 * Partition earlier sightings (all strictly before today, already sorted
 * newest-first by the API) into calendar buckets: "This week" (since the most
 * recent Monday), "This month" (earlier this calendar month but before this
 * week), and "Earlier" (anything older). Empty buckets are dropped so no
 * heading renders without cards. Each item keeps its index in the input list
 * for lightbox navigation.
 */
function buildEarlierGroups(list: Detection[]): EarlierGroup[] {
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const monthStart = new Date(now.getFullYear(), now.getMonth(), 1);
  // Week starts on Monday; getDay() is 0=Sun..6=Sat.
  const daysSinceMonday = (todayStart.getDay() + 6) % 7;
  const weekStart = new Date(todayStart);
  weekStart.setDate(todayStart.getDate() - daysSinceMonday);

  const thisWeek: EarlierGroup["items"] = [];
  const thisMonth: EarlierGroup["items"] = [];
  const older: EarlierGroup["items"] = [];
  list.forEach((detection, index) => {
    // Naive-local ISO strings parse as local time, matching the boundaries.
    const ts = new Date(detection.timestamp);
    if (ts >= weekStart) {
      thisWeek.push({ detection, index });
    } else if (ts >= monthStart) {
      thisMonth.push({ detection, index });
    } else {
      older.push({ detection, index });
    }
  });

  return [
    { label: "This week", items: thisWeek },
    { label: "This month", items: thisMonth },
    { label: "Earlier", items: older },
  ].filter((group) => group.items.length > 0);
}

export function Dashboard() {
  const [today, setToday] = useState<Detection[]>([]);
  const [earlier, setEarlier] = useState<Detection[]>([]);
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
    // Local midnight is the boundary between "today" and "earlier". The two
    // queries are disjoint: today is everything at/after midnight, earlier is
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
        limit: EARLIER_LIMIT,
        min_confidence: minConf,
      }),
    ])
      .then(([todayData, earlierData]) => {
        setToday(todayData);
        setEarlier(earlierData);
        setError(null);
        setLoading(false);
        setLightbox(null);
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : "Failed to load detections");
        setLoading(false);
      });
  }, [minConfidence]);

  const activeList = lightbox?.section === "today" ? today : earlier;
  const currentDetection = lightbox !== null ? activeList[lightbox.index] ?? null : null;

  const removeDetection = (deletedId: number): void => {
    setToday((prev) => prev.filter((d) => d.id !== deletedId));
    setEarlier((prev) => prev.filter((d) => d.id !== deletedId));
    setLightbox(null);
  };

  // Replace the corrected detection in place in both strips, keeping the
  // lightbox open on it so its species/reference refresh live.
  const updateDetection = (updated: Detection): void => {
    setToday((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
    setEarlier((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
  };

  const noConfidenceSuffix =
    minConfidence > 0 ? ` at ${minConfidence}% match or better` : "";

  /** Render the "Spotted today" horizontal strip (or its empty/loading state). */
  const renderTodayStrip = () => {
    if (loading) {
      return <p className="text-sm text-bark animate-pulse">Checking the feeder…</p>;
    }
    if (today.length === 0) {
      return <p className="text-sm text-bark">{`No birds spotted today${noConfidenceSuffix}.`}</p>;
    }
    return (
      <div className="flex gap-4 overflow-x-auto pb-3">
        {today.map((d, i) => (
          <DetectionCard
            key={d.id}
            detection={d}
            onOpenLightbox={() => setLightbox({ section: "today", index: i })}
          />
        ))}
      </div>
    );
  };

  /**
   * Render the earlier sightings as a grid, split into time-bucketed sections
   * ("This week" / "This month" / "Earlier"). Falls back to a single titled
   * loading/empty section before any cards exist.
   */
  const renderEarlierSections = () => {
    if (loading) {
      return (
        <section className="space-y-4">
          <h2 className="eyebrow">Earlier sightings</h2>
          <p className="text-sm text-bark animate-pulse">Checking the feeder…</p>
        </section>
      );
    }
    if (earlier.length === 0) {
      return (
        <section className="space-y-4">
          <h2 className="eyebrow">Earlier sightings</h2>
          <p className="text-sm text-bark">{`Nothing earlier to show${noConfidenceSuffix}.`}</p>
        </section>
      );
    }
    return buildEarlierGroups(earlier).map((group) => (
      <section key={group.label} className="space-y-4">
        <h2 className="eyebrow">{group.label}</h2>
        <div className="flex flex-wrap gap-4">
          {group.items.map(({ detection, index }) => (
            <DetectionCard
              key={detection.id}
              detection={detection}
              onOpenLightbox={() => setLightbox({ section: "earlier", index })}
            />
          ))}
        </div>
      </section>
    ));
  };

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

      {/* Minimum confidence slider applies to every strip below. */}
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
        {renderTodayStrip()}
      </section>

      {renderEarlierSections()}
    </div>
  );
}
