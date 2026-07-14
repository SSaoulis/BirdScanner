import { useEffect, useState } from "react";
import { api, formatUptime, type SystemStatus } from "../api";

const POLL_INTERVAL_MS = 5000;

function gaugeColor(percent: number): string {
  if (percent >= 80) return "bg-rust";
  if (percent >= 60) return "bg-gold";
  return "bg-sage";
}

function tempColor(celsius: number): string {
  if (celsius >= 70) return "bg-rust";
  if (celsius >= 50) return "bg-gold";
  return "bg-sage";
}

interface GaugeBarProps {
  label: string;
  value: number | null;
  unit: string;
  max: number;
  colorClass: string;
  min?: number;
}

function GaugeBar({ label, value, unit, max, colorClass, min = 0 }: GaugeBarProps) {
  const pct =
    value === null ? 0 : Math.min(Math.max(((value - min) / (max - min)) * 100, 0), 100);
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-bark">{label}</span>
        <span className="tnum font-semibold text-ink">
          {value === null ? "—" : `${value.toFixed(1)}${unit}`}
        </span>
      </div>
      <div className="h-2 rounded-full bg-paper overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${colorClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function SystemMonitor() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchStatus() {
      try {
        const s = await api.system.get();
        if (!cancelled) {
          setStatus(s);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load system status");
      }
    }

    fetchStatus();
    const id = setInterval(fetchStatus, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="rounded-2xl border border-line bg-card p-5 shadow-plate space-y-4">
      <div>
        <p className="eyebrow">The station</p>
        <p className="mt-1 text-sm text-bark">How the Pi behind the feeder is holding up</p>
      </div>

      {error && (
        <p className="text-sm text-rust">{error}</p>
      )}

      {/*
        The gauge structure is always rendered (with placeholder dashes/empty
        bars until the first poll resolves) so the component keeps a fixed size
        from the initial render — this avoids the page shifting when values
        arrive. When `status` is null, every gauge falls back to a null value.
      */}
      <div className="space-y-3" aria-busy={status === null && !error}>
        <GaugeBar
          label="CPU"
          value={status?.cpu_percent ?? null}
          unit="%"
          max={100}
          colorClass={gaugeColor(status?.cpu_percent ?? 0)}
        />
        <GaugeBar
          label="Memory"
          value={status?.memory_percent ?? null}
          unit="%"
          max={100}
          colorClass={gaugeColor(status?.memory_percent ?? 0)}
        />
        <GaugeBar
          label="Disk"
          value={status?.disk_percent ?? null}
          unit="%"
          max={100}
          colorClass={gaugeColor(status?.disk_percent ?? 0)}
        />
        <GaugeBar
          label="CPU Temp"
          value={status?.cpu_temp_celsius ?? null}
          unit="°C"
          min={25}
          max={85}
          colorClass={tempColor(status?.cpu_temp_celsius ?? 0)}
        />
        <div className="flex justify-between text-sm pt-2 border-t border-line">
          <span className="text-bark">Watching for</span>
          <span className="tnum font-semibold text-ink">
            {status === null ? "—" : formatUptime(status.uptime_seconds)}
          </span>
        </div>
      </div>
    </div>
  );
}
