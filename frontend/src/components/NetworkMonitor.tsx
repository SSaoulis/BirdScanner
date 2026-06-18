import { useEffect, useRef, useState } from "react";
import {
  api,
  type NetworkHistory,
  type NetworkRange,
  type NetworkSample,
  type SpeedTestResult,
} from "../api";

const POLL_INTERVAL_MS = 3000;

const RANGES: { value: NetworkRange; label: string; seconds: number }[] = [
  { value: "5m", label: "5 min", seconds: 300 },
  { value: "30m", label: "30 min", seconds: 1800 },
  { value: "1h", label: "1 hour", seconds: 3600 },
];

// Internal SVG coordinate space; the element scales to its container via width.
const VIEW_W = 640;
const VIEW_H = 180;
const PAD = { top: 12, right: 8, bottom: 18, left: 44 };
const PLOT_W = VIEW_W - PAD.left - PAD.right;
const PLOT_H = VIEW_H - PAD.top - PAD.bottom;

/** Format a kilobits/sec rate adaptively as Kbps or Mbps. */
function formatRate(kbps: number): string {
  if (kbps >= 1000) return `${(kbps / 1000).toFixed(1)} Mbps`;
  return `${kbps.toFixed(0)} Kbps`;
}

/** Round a value up to a "nice" axis maximum so the y-scale isn't jittery. */
function niceCeil(value: number): number {
  if (value <= 0) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(value)));
  const norm = value / pow;
  const step = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
  return step * pow;
}

interface ChartProps {
  samples: NetworkSample[];
  windowSeconds: number;
}

/**
 * Hand-rolled SVG line chart of download (rx) and upload (tx) throughput.
 *
 * The x-axis is anchored to the live window ``[now - windowSeconds, now]`` so
 * the trace grows in from the right as history accumulates rather than
 * stretching two early samples across the full width. The y-axis auto-scales to
 * a "nice" ceiling above the peak rate in the window.
 */
function Chart({ samples, windowSeconds }: ChartProps) {
  if (samples.length === 0) {
    return (
      <div className="flex h-44 items-center justify-center text-sm text-bark">
        Gathering samples…
      </div>
    );
  }

  const now = samples[samples.length - 1].t;
  const windowStart = now - windowSeconds;
  const peak = Math.max(
    ...samples.map((s) => Math.max(s.rx_kbps, s.tx_kbps)),
    1,
  );
  const yMax = niceCeil(peak);

  const xFor = (t: number): number => {
    const frac = Math.min(Math.max((t - windowStart) / windowSeconds, 0), 1);
    return PAD.left + frac * PLOT_W;
  };
  const yFor = (v: number): number => PAD.top + (1 - v / yMax) * PLOT_H;

  const line = (key: "rx_kbps" | "tx_kbps"): string =>
    samples.map((s, i) => `${i === 0 ? "M" : "L"}${xFor(s.t).toFixed(1)},${yFor(s[key]).toFixed(1)}`).join(" ");

  const area = (key: "rx_kbps" | "tx_kbps"): string => {
    const baseline = yFor(0);
    const start = `M${xFor(samples[0].t).toFixed(1)},${baseline.toFixed(1)}`;
    const tops = samples.map((s) => `L${xFor(s.t).toFixed(1)},${yFor(s[key]).toFixed(1)}`).join(" ");
    const end = `L${xFor(samples[samples.length - 1].t).toFixed(1)},${baseline.toFixed(1)} Z`;
    return `${start} ${tops} ${end}`;
  };

  // Three horizontal gridlines (0, mid, max) with labels.
  const ticks = [0, yMax / 2, yMax];

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className="w-full"
      role="img"
      aria-label="Network throughput over time"
      preserveAspectRatio="none"
    >
      {ticks.map((v) => (
        <g key={v}>
          <line
            x1={PAD.left}
            x2={PAD.left + PLOT_W}
            y1={yFor(v)}
            y2={yFor(v)}
            className="stroke-line"
            strokeWidth={1}
          />
          <text
            x={PAD.left - 6}
            y={yFor(v) + 3}
            textAnchor="end"
            className="fill-bark"
            style={{ fontSize: "9px" }}
          >
            {formatRate(v)}
          </text>
        </g>
      ))}

      {/* Download (rx) — sage green */}
      <path d={area("rx_kbps")} className="fill-sage/15" />
      <path
        d={line("rx_kbps")}
        fill="none"
        className="stroke-sage-deep"
        strokeWidth={1.8}
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* Upload (tx) — goldfinch ochre */}
      <path d={area("tx_kbps")} className="fill-gold/15" />
      <path
        d={line("tx_kbps")}
        fill="none"
        className="stroke-gold-deep"
        strokeWidth={1.8}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Coloured legend swatch + label. */
function Legend({ colorClass, label, value }: { colorClass: string; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className={`inline-block h-2.5 w-2.5 rounded-full ${colorClass}`} />
      <span className="text-bark">{label}</span>
      <span className="tnum font-semibold text-ink">{value}</span>
    </div>
  );
}

/**
 * Network panel for the Hardware tab.
 *
 * Top half: a passive usage graph of the Pi's NIC download/upload throughput,
 * polled every few seconds with a 5m / 30m / 1h window toggle. Reading the
 * counters costs no bandwidth, so it runs continuously while the page is open.
 *
 * Bottom half: an on-demand internet speed test. Each run downloads ~1 MB and
 * uploads ~256 KB against a public endpoint, so it is a manual button (never
 * polled) to respect the Pi's limited connection.
 */
export function NetworkMonitor() {
  const [range, setRange] = useState<NetworkRange>("5m");
  const [history, setHistory] = useState<NetworkHistory | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<SpeedTestResult | null>(null);
  const [testError, setTestError] = useState<string | null>(null);

  // `range` is read inside the polling closure; a ref keeps the interval from
  // being torn down and recreated on every range change.
  const rangeRef = useRef(range);
  rangeRef.current = range;

  useEffect(() => {
    let cancelled = false;

    async function fetchHistory() {
      try {
        const h = await api.network.history(rangeRef.current);
        if (!cancelled) {
          setHistory(h);
          setHistoryError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setHistoryError(e instanceof Error ? e.message : "Failed to load network history");
        }
      }
    }

    fetchHistory();
    const id = setInterval(fetchHistory, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Refetch immediately when the window changes rather than waiting for the
  // next poll tick, so the toggle feels responsive.
  useEffect(() => {
    let cancelled = false;
    api.network
      .history(range)
      .then((h) => {
        if (!cancelled) setHistory(h);
      })
      .catch(() => {
        /* the polling effect surfaces errors; ignore here */
      });
    return () => {
      cancelled = true;
    };
  }, [range]);

  async function runSpeedTest() {
    setTesting(true);
    setTestError(null);
    try {
      const r = await api.network.speedTest();
      setResult(r);
    } catch (e) {
      setTestError(e instanceof Error ? e.message : "Speed test failed");
    } finally {
      setTesting(false);
    }
  }

  const samples = history?.samples ?? [];
  const latest = samples.length > 0 ? samples[samples.length - 1] : null;
  const windowSeconds = RANGES.find((r) => r.value === range)?.seconds ?? 300;

  const rangeButtonClass = (active: boolean): string =>
    `px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
      active ? "bg-gold text-card shadow-sm" : "border border-line bg-card text-bark hover:text-ink"
    }`;

  return (
    <div className="space-y-6">
      {/* ── Usage graph ─────────────────────────────────────────────── */}
      <div className="rounded-2xl border border-line bg-card p-5 shadow-plate space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="eyebrow">Traffic at the perch</p>
            <p className="mt-1 text-sm text-bark">
              What the Pi is sending and receiving right now
            </p>
          </div>
          <div className="flex gap-2">
            {RANGES.map((r) => (
              <button
                key={r.value}
                onClick={() => setRange(r.value)}
                className={rangeButtonClass(range === r.value)}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
          <Legend
            colorClass="bg-sage-deep"
            label="Download"
            value={latest ? formatRate(latest.rx_kbps) : "—"}
          />
          <Legend
            colorClass="bg-gold-deep"
            label="Upload"
            value={latest ? formatRate(latest.tx_kbps) : "—"}
          />
        </div>

        {historyError ? (
          <p className="text-sm text-rust">{historyError}</p>
        ) : (
          <Chart samples={samples} windowSeconds={windowSeconds} />
        )}
      </div>

      {/* ── Speed test ──────────────────────────────────────────────── */}
      <div className="rounded-2xl border border-line bg-card p-5 shadow-plate space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="eyebrow">Reach to the wider world</p>
            <p className="mt-1 text-sm text-bark">
              On-demand internet speed test — uses ~1 MB of data per run
            </p>
          </div>
          <button
            onClick={runSpeedTest}
            disabled={testing}
            className="rounded-lg bg-gold px-4 py-2 text-sm font-semibold text-card shadow-sm transition-colors hover:bg-gold-deep disabled:cursor-not-allowed disabled:opacity-50"
          >
            {testing ? "Testing…" : "Test network"}
          </button>
        </div>

        {testError && (
          <div className="rounded-lg border border-rust/40 bg-rust/10 px-4 py-3 text-sm text-rust">
            {testError}
          </div>
        )}

        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-xl border border-line bg-paper p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-bark">Download</p>
            <p className="tnum mt-1 text-2xl font-semibold text-sage-deep">
              {result ? result.download_mbps.toFixed(1) : "—"}
              <span className="ml-1 text-sm font-normal text-bark">Mbps</span>
            </p>
          </div>
          <div className="rounded-xl border border-line bg-paper p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-bark">Upload</p>
            <p className="tnum mt-1 text-2xl font-semibold text-gold-deep">
              {result ? result.upload_mbps.toFixed(1) : "—"}
              <span className="ml-1 text-sm font-normal text-bark">Mbps</span>
            </p>
          </div>
        </div>

        {result && (
          <p className="text-xs text-bark">
            Last tested {new Date(result.ran_at * 1000).toLocaleTimeString()} ·{" "}
            {(result.download_bytes / 1024).toFixed(0)} KB down,{" "}
            {(result.upload_bytes / 1024).toFixed(0)} KB up
          </p>
        )}
      </div>
    </div>
  );
}
