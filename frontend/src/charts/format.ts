/** Small formatting helpers shared by the Statistics charts. */

/** Format minutes-since-midnight as a short clock label (e.g. 390 → "6:30"). */
export function minuteToClock(minute: number): string {
  const h = Math.floor(minute / 60);
  const m = minute % 60;
  return `${h}:${String(m).padStart(2, "0")}`;
}

/** Format minutes-since-midnight as a compact hour label (e.g. 0 → "12a", 780 → "1p"). */
export function hourLabel(minute: number): string {
  const h = Math.round(minute / 60) % 24;
  if (h === 0) return "12a";
  if (h === 12) return "12p";
  return h < 12 ? `${h}a` : `${h - 12}p`;
}

/** Two-digit hour string for heatmap columns (e.g. 9 → "09"). */
export function hourColumn(hour: number): string {
  return String(hour).padStart(2, "0");
}

/** Shorten a date bucket label (`YYYY-MM-DD` → "Jun 3"; weeks pass through). */
export function shortDate(label: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(label);
  if (!match) return label; // week buckets ("YYYY-WW") pass through unchanged
  const [, y, m, d] = match;
  const date = new Date(Number(y), Number(m) - 1, Number(d));
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/**
 * Pick up to `count` evenly-spaced values from `values` (always keeps the last),
 * for thinning a crowded categorical axis.
 */
export function evenlySpaced<T>(values: T[], count: number): T[] {
  if (values.length <= count) return values;
  const step = (values.length - 1) / (count - 1);
  const out: T[] = [];
  for (let i = 0; i < count; i++) out.push(values[Math.round(i * step)]);
  return Array.from(new Set(out));
}

/** Weekday labels (Monday-first) for the activity heatmap rows. */
export const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
