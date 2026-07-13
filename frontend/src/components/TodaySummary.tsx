import { type Detection } from "../api";

/** Today's aggregate counts derived from the day's detections. */
interface TodayCounts {
  /** Number of sightings today. */
  total: number;
  /** Number of distinct species seen today. */
  distinctSpecies: number;
  /** The species seen most often today, or null when there are none. */
  mostCommon: string | null;
}

/**
 * Compute the day's aggregate counts (total sightings, distinct species, and
 * the most-frequent species) from the list of today's detections. Ties on the
 * most-common species resolve to whichever the reduce encounters first.
 */
function todayCounts(today: Detection[]): TodayCounts {
  const bySpecies = new Map<string, number>();
  for (const d of today) {
    bySpecies.set(d.species, (bySpecies.get(d.species) ?? 0) + 1);
  }
  let mostCommon: string | null = null;
  let best = 0;
  for (const [species, count] of bySpecies) {
    if (count > best) {
      best = count;
      mostCommon = species;
    }
  }
  return { total: today.length, distinctSpecies: bySpecies.size, mostCommon };
}

/** One labelled figure in the today-counts strip. */
function CountTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <p className="eyebrow">{label}</p>
      <p
        className="tnum max-w-[14rem] truncate font-display text-lg font-medium leading-none text-ink"
        title={value}
      >
        {value}
      </p>
    </div>
  );
}

interface TodaySummaryProps {
  /** Today's detections, used to compute the aggregate counts. */
  today: Detection[];
}

/**
 * A small at-a-glance summary panel of the day's totals — total sightings,
 * distinct species, and the most common species — shown directly beneath the
 * Dashboard title. Purely presentational; the Dashboard owns the data.
 */
export function TodaySummary({ today }: TodaySummaryProps) {
  const counts = todayCounts(today);

  return (
    <section className="rounded-xl border border-line bg-card p-4 shadow-plate">
      <div className="flex flex-wrap gap-x-8 gap-y-3">
        <CountTile label="Sightings today" value={String(counts.total)} />
        <CountTile label="Species today" value={String(counts.distinctSpecies)} />
        <CountTile label="Most common" value={counts.mostCommon ?? "—"} />
      </div>
    </section>
  );
}
