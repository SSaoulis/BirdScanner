import { useEffect, useState } from "react";
import { api, formatUptime, type SystemStatus } from "../api";

const POLL_INTERVAL_MS = 5000;

function gaugeColor(percent: number): string {
  if (percent >= 80) return "bg-red-500";
  if (percent >= 60) return "bg-yellow-400";
  return "bg-emerald-500";
}

function tempColor(celsius: number): string {
  if (celsius >= 70) return "bg-red-500";
  if (celsius >= 50) return "bg-yellow-400";
  return "bg-emerald-500";
}

interface GaugeBarProps {
  label: string;
  value: number | null;
  unit: string;
  max: number;
  colorClass: string;
}

function GaugeBar({ label, value, unit, max, colorClass }: GaugeBarProps) {
  const pct = value === null ? 0 : Math.min((value / max) * 100, 100);
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono font-semibold text-white">
          {value === null ? "—" : `${value.toFixed(1)}${unit}`}
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-700 overflow-hidden">
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
    <div className="bg-slate-800 rounded-2xl p-5 space-y-4">
      <h2 className="text-lg font-semibold text-white">System</h2>

      {error && (
        <p className="text-sm text-red-400">{error}</p>
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
          max={100}
          colorClass={tempColor(status?.cpu_temp_celsius ?? 0)}
        />
        <div className="flex justify-between text-sm pt-1 border-t border-slate-700">
          <span className="text-slate-400">Uptime</span>
          <span className="font-mono font-semibold text-white">
            {status === null ? "—" : formatUptime(status.uptime_seconds)}
          </span>
        </div>
      </div>
    </div>
  );
}
