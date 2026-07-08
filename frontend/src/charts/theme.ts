/**
 * Shared Nivo theme + palette for the Statistics charts.
 *
 * The design system is the "field journal" light-only theme (see
 * `tailwind.config.ts`). Nivo renders inline SVG styles, so — unlike the
 * hand-rolled SVG charts — it can't consume Tailwind utility classes; the token
 * hexes are threaded through this `theme` object and the palette arrays instead.
 *
 * The categorical palette below was validated for colour-vision-deficiency
 * separation with the dataviz skill's validator (6 series: worst-pair ΔE ≈ 12.7,
 * all above the chroma floor and ≥ 3:1 contrast on the card surface). Six is the
 * CVD-safe ceiling for this warm palette, so the timeline requests `top = 6` and
 * folds the rest into a neutral "Other".
 */

// --- design tokens (mirrors tailwind.config.ts) ---------------------------
const INK = "#2C3A2E";
const BARK = "#6E6448";
const LINE = "#D8CDB0";
const CARD = "#F6F1E2";
const SAGE_DEEP = "#566048";

/** Ordered categorical hues for species series (green, rust, blue, ochre, teal, plum). */
export const CATEGORICAL = [
  "#4C7A3F",
  "#B23A24",
  "#34589C",
  "#9A7B12",
  "#00998C",
  "#8A3A6B",
] as const;

/** Neutral colour for the aggregated "Other" species bucket (recedes on purpose). */
export const OTHER_COLOR = "#8C8069";
/** The literal key the backend uses for the folded-together species bucket. */
export const OTHER_KEY = "Other";

/** Single-series accent for one-measure charts (diversity, density, bars). */
export const ACCENT = SAGE_DEEP;
/** Two-tone pair for the daily activity window: sunrise (ochre) → dusk (blue). */
export const SUNRISE = "#9A7B12";
export const DUSK = "#34589C";

/**
 * Return the stable colour for a species series key.
 *
 * Colour follows the species' slot in the (count-ordered) `keys` list returned
 * by the API, so every series in one response gets a distinct hue; the ``Other``
 * bucket always takes the neutral colour.
 *
 * @param key - The series key (a species name or ``"Other"``).
 * @param keys - The ordered list of species keys for the current response.
 * @returns A hex colour string.
 */
export function seriesColor(key: string, keys: string[]): string {
  if (key === OTHER_KEY) return OTHER_COLOR;
  const idx = keys.indexOf(key);
  return CATEGORICAL[((idx < 0 ? 0 : idx) % CATEGORICAL.length)];
}

/**
 * Interpolate a single-hue sequential colour (pale → deep sage) for the heatmap.
 *
 * @param t - Normalised magnitude in [0, 1].
 * @returns A hex colour string; higher `t` is darker/greener.
 */
export function sequentialColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const from = [0xef, 0xea, 0xd9]; // near-paper light end
  const to = [0x3f, 0x4a, 0x34]; // deep sage
  const ch = (i: number): number => Math.round(from[i] + (to[i] - from[i]) * clamped);
  return `#${[ch(0), ch(1), ch(2)].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

/** Empty-cell colour for the heatmap (a hair above the card surface). */
export const HEATMAP_EMPTY = "#F0EBDA";

/**
 * Circular Gaussian smoothing for a time-of-day histogram.
 *
 * The bins wrap around the 24-hour clock (23:59 → 00:00), so the kernel indexes
 * modulo the array length — this turns the raw counts into a smooth density
 * curve without a discontinuity at midnight.
 *
 * @param values - Per-bin counts, evenly spaced around 24 hours.
 * @param sigma - Kernel width in bins (larger = smoother).
 * @returns The smoothed values (same length as the input).
 */
export function circularGaussianSmooth(values: number[], sigma = 1.6): number[] {
  const n = values.length;
  if (n === 0) return [];
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel: number[] = [];
  for (let k = -radius; k <= radius; k++) {
    kernel.push(Math.exp(-(k * k) / (2 * sigma * sigma)));
  }
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    let wsum = 0;
    for (let k = -radius; k <= radius; k++) {
      const w = kernel[k + radius];
      const idx = (((i + k) % n) + n) % n;
      acc += values[idx] * w;
      wsum += w;
    }
    out[i] = acc / wsum;
  }
  return out;
}

/** Nivo `theme` object mapping the field-journal tokens onto chart chrome. */
export const nivoTheme = {
  background: "transparent",
  text: {
    fontFamily: '"Hanken Grotesk", system-ui, sans-serif',
    fontSize: 12,
    fill: BARK,
  },
  axis: {
    domain: { line: { stroke: LINE, strokeWidth: 1 } },
    ticks: {
      line: { stroke: LINE, strokeWidth: 1 },
      text: { fill: BARK, fontSize: 11 },
    },
    legend: {
      text: {
        fill: INK,
        fontSize: 12,
        fontFamily: '"Hanken Grotesk", system-ui, sans-serif',
      },
    },
  },
  grid: { line: { stroke: LINE, strokeWidth: 1, strokeDasharray: "2 4" } },
  legends: { text: { fill: INK, fontSize: 12 } },
  labels: { text: { fill: INK, fontSize: 11 } },
  tooltip: {
    container: {
      background: CARD,
      color: INK,
      fontSize: 12,
      fontFamily: '"Hanken Grotesk", system-ui, sans-serif',
      border: `1px solid ${LINE}`,
      borderRadius: 8,
      boxShadow: "0 6px 16px rgba(44,58,46,0.16)",
    },
  },
};
