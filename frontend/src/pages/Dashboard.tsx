import { useEffect, useState } from "react";
import { api, type Detection } from "../api";
import { DetectionCard } from "../components/DetectionCard";
import { ExpectedThisWeek } from "../components/ExpectedThisWeek";
import { FeaturedDetection } from "../components/FeaturedDetection";
import { Lightbox } from "../components/Lightbox";
import { TodaySummary } from "../components/TodaySummary";

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

    Promise.all([
      api.detections.list({
        from: toNaiveISO(todayStart),
        limit: TODAY_LIMIT,
      }),
      api.detections.list({
        to: toNaiveISO(beforeToday),
        limit: EARLIER_LIMIT,
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
  }, []);

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

  /** Render the "Spotted today" horizontal strip (or its empty/loading state). */
  const renderTodayStrip = () => {
    if (loading) {
      return <p className="text-sm text-bark animate-pulse">Checking the feeder…</p>;
    }
    if (today.length === 0) {
      return <p className="text-sm text-bark">No birds spotted today.</p>;
    }
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
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
          <p className="text-sm text-bark">Nothing earlier to show.</p>
        </section>
      );
    }
    return buildEarlierGroups(earlier).map((group) => (
      <section key={group.label} className="space-y-4">
        <h2 className="eyebrow">{group.label}</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
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
    <div className="mx-auto max-w-6xl px-6 py-8">
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
          position={{ index: lightbox.index, total: activeList.length }}
          prevDetection={lightbox.index > 0 ? activeList[lightbox.index - 1] ?? null : null}
          nextDetection={
            lightbox.index < activeList.length - 1
              ? activeList[lightbox.index + 1] ?? null
              : null
          }
        />
      )}

      <header className="mb-6">
        <p className="eyebrow mb-2">At the feeder</p>
        <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">
          Today&rsquo;s visitors
        </h1>
        <p className="mt-1 text-sm text-bark">{todayLabel}</p>
      </header>

      {/* Day-at-a-glance totals, directly beneath the title. Rendered once loaded
          so it doesn't flash zeros; shows 0 / 0 / — when nothing's been spotted. */}
      {!loading && (
        <div className="mb-8">
          <TodaySummary today={today} />
        </div>
      )}

      {/* Feed on the left, the "In season" marginalia note in a narrow right rail.
          The rail is source-first so it stacks under the header on mobile (a useful
          glance before scrolling the feed) but pins to the right column on desktop. */}
      <div className="lg:grid lg:grid-cols-[1fr_15rem] lg:items-start lg:gap-8">
        <aside className="mb-8 lg:col-start-2 lg:mb-0 lg:sticky lg:top-8">
          {/* What the geomodel expects around the feeder this time of year. */}
          <ExpectedThisWeek />
        </aside>

        <div className="min-w-0 space-y-8 lg:col-start-1 lg:row-start-1">
          {error && <p className="text-sm text-rust">{error}</p>}

          {/* Hero: the most recent sighting at full size + its stats. Hidden
              when nothing has been spotted today (or filtered out). */}
          {!loading && today.length > 0 && (
            <FeaturedDetection
              detection={today[0]}
              onOpenLightbox={() => setLightbox({ section: "today", index: 0 })}
            />
          )}

          <section className="space-y-4">
            <h2 className="eyebrow">Spotted today</h2>
            {renderTodayStrip()}
          </section>

          {renderEarlierSections()}
        </div>
      </div>
    </div>
  );
}
