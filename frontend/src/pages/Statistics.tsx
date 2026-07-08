import { useEffect, useMemo, useState } from "react";

import {
  api,
  type ActivityResponse,
  type DailyWindowResponse,
  type FirstSighting,
  type SpeciesSummary,
  type StatsInterval,
  type StatsParams,
  type StatsRange,
  type StatsSummary,
  type TimeOfDay,
  type TimelineResponse,
} from "../api";
import { ActivityHeatmap } from "../components/charts/ActivityHeatmap";
import { DailyActivityWindow } from "../components/charts/DailyActivityWindow";
import { MostCommonSpecies } from "../components/charts/MostCommonSpecies";
import { NewSpeciesTimeline } from "../components/charts/NewSpeciesTimeline";
import { SightingsOverTime } from "../components/charts/SightingsOverTime";
import { SpeciesDiversity } from "../components/charts/SpeciesDiversity";
import { TimeOfDayDensity } from "../components/charts/TimeOfDayDensity";

const RANGES: { value: StatsRange; label: string; days: number | null }[] = [
  { value: "7d", label: "7 days", days: 7 },
  { value: "30d", label: "30 days", days: 30 },
  { value: "90d", label: "90 days", days: 90 },
  { value: "all", label: "All time", days: null },
];

const INTERVALS: { value: StatsInterval; label: string }[] = [
  { value: "day", label: "Daily" },
  { value: "week", label: "Weekly" },
];

/** The top-N species kept as their own stack series (CVD-safe ceiling; see charts/theme). */
const TIMELINE_TOP = 6;

/** Format a Date as a naive-local ISO string (no timezone), matching stored timestamps. */
function toNaiveISO(d: Date): string {
  const p = (n: number): string => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}` +
    `T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
  );
}

/** Derive the `from` cutoff for a range choice (naive local; `all` omits it). */
function rangeParams(range: StatsRange): StatsParams {
  const days = RANGES.find((r) => r.value === range)?.days ?? null;
  if (days === null) return {};
  const from = new Date();
  from.setDate(from.getDate() - days);
  from.setHours(0, 0, 0, 0);
  return { from: toNaiveISO(from) };
}

const toggleClass = (active: boolean): string =>
  `px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
    active
      ? "bg-gold text-card shadow-sm"
      : "border border-line bg-card text-bark hover:text-ink"
  }`;

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-line bg-card px-4 py-3 shadow-plate">
      <p className="eyebrow mb-1">{label}</p>
      <p className="tnum font-display text-2xl font-semibold text-ink">{value}</p>
    </div>
  );
}

/**
 * Statistics page.
 *
 * Aggregate temporal + species-composition charts over the detection history,
 * driven by a global time-range selector (7d/30d/90d/all) and a day/week
 * interval toggle. Data comes from the `/api/stats/*` endpoints (server-side
 * `GROUP BY`); the new-species curve and the most-common leaderboard are
 * all-time by nature. Charts are Nivo (see `charts/theme.ts`).
 */
export function Statistics() {
  const [range, setRange] = useState<StatsRange>("30d");
  const [interval, setInterval] = useState<StatsInterval>("day");
  const [todSpecies, setTodSpecies] = useState<string>("");

  const [summary, setSummary] = useState<StatsSummary | null>(null);
  const [activity, setActivity] = useState<ActivityResponse | null>(null);
  const [dailyWindow, setDailyWindow] = useState<DailyWindowResponse | null>(null);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [timeOfDay, setTimeOfDay] = useState<TimeOfDay | null>(null);
  const [speciesList, setSpeciesList] = useState<SpeciesSummary[] | null>(null);
  const [firstSightings, setFirstSightings] = useState<FirstSighting[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const params = useMemo(() => rangeParams(range), [range]);

  // All-time datasets: fetched once on mount (range-independent).
  useEffect(() => {
    let cancelled = false;
    Promise.all([api.species.list(), api.stats.firstSightings()])
      .then(([species, sightings]) => {
        if (cancelled) return;
        setSpeciesList(species);
        setFirstSightings(sightings);
      })
      .catch((e: unknown) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, []);

  // Range-driven datasets (summary + activity + daily window).
  useEffect(() => {
    let cancelled = false;
    Promise.all([
      api.stats.summary(params),
      api.stats.activity(params),
      api.stats.dailyWindow(params),
    ])
      .then(([s, a, d]) => {
        if (cancelled) return;
        setSummary(s);
        setActivity(a);
        setDailyWindow(d);
      })
      .catch((e: unknown) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [params]);

  // Timeline depends on the interval as well as the range.
  useEffect(() => {
    let cancelled = false;
    api.stats
      .timeline({ ...params, interval, top: TIMELINE_TOP })
      .then((t) => !cancelled && setTimeline(t))
      .catch((e: unknown) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [params, interval]);

  // Time-of-day depends on the optional species filter as well as the range.
  useEffect(() => {
    let cancelled = false;
    api.stats
      .timeOfDay({ ...params, species: todSpecies || undefined })
      .then((t) => !cancelled && setTimeOfDay(t))
      .catch((e: unknown) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [params, todSpecies]);

  const ready =
    summary &&
    activity &&
    dailyWindow &&
    timeline &&
    timeOfDay &&
    speciesList &&
    firstSightings;

  const speciesControl = (
    <select
      value={todSpecies}
      onChange={(e) => setTodSpecies(e.target.value)}
      className="rounded-lg border border-line bg-paper px-2 py-1 text-xs text-ink focus:border-gold focus:outline-none focus:ring-2 focus:ring-gold"
      aria-label="Filter time-of-day by species"
    >
      <option value="">All species</option>
      {(speciesList ?? []).map((s) => (
        <option key={s.species} value={s.species}>
          {s.species}
        </option>
      ))}
    </select>
  );

  return (
    <div className="mx-auto max-w-6xl px-6 py-8 space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="eyebrow mb-2">Patterns in the record</p>
          <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">
            Statistics
          </h1>
          <p className="mt-1 text-sm text-bark">
            When and what the feeder attracts, over time
          </p>
        </div>
        <div className="flex gap-2">
          {RANGES.map((r) => (
            <button
              key={r.value}
              type="button"
              onClick={() => setRange(r.value)}
              className={toggleClass(range === r.value)}
            >
              {r.label}
            </button>
          ))}
        </div>
      </header>

      {error && (
        <div className="rounded-xl border border-line bg-card px-4 py-3 text-sm text-rust">
          Couldn't load statistics: {error}
        </div>
      )}

      {!ready ? (
        <div className="flex h-64 items-center justify-center text-sm text-bark">
          Gathering the record…
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatTile label="Detections" value={summary.total.toLocaleString()} />
            <StatTile label="Species" value={String(summary.distinct_species)} />
            <StatTile label="Corrected" value={String(summary.corrected_count)} />
            <StatTile
              label="Total known"
              value={String(firstSightings.length)}
            />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="lg:col-span-2">
              <SightingsOverTime timeline={timeline} />
            </div>

            <div className="flex items-center gap-2 lg:col-span-2">
              <span className="eyebrow">Interval</span>
              {INTERVALS.map((iv) => (
                <button
                  key={iv.value}
                  type="button"
                  onClick={() => setInterval(iv.value)}
                  className={toggleClass(interval === iv.value)}
                >
                  {iv.label}
                </button>
              ))}
            </div>

            <TimeOfDayDensity
              bins={timeOfDay.bins}
              binMinutes={timeOfDay.bin_minutes}
              speciesControl={speciesControl}
            />
            <ActivityHeatmap cells={activity.cells} />
            <SpeciesDiversity timeline={timeline} />
            <MostCommonSpecies species={speciesList} />
            <DailyActivityWindow days={dailyWindow.days} />
            <NewSpeciesTimeline sightings={firstSightings} />
          </div>
        </>
      )}
    </div>
  );
}
