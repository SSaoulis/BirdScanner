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
  value: number;
  unit: string;
  max: number;
  colorClass: string;
}

function GaugeBar({ label, value, unit, max, colorClass }: GaugeBarProps) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono font-semibold text-white">
          {value.toFixed(1)}{unit}
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

      {status === null && !error && (
        <p className="text-sm text-slate-500 animate-pulse">Loading…</p>
      )}

      {status && (
        <div className="space-y-3">
          <GaugeBar
            label="CPU"
            value={status.cpu_percent}
            unit="%"
            max={100}
            colorClass={gaugeColor(status.cpu_percent)}
          />
          <GaugeBar
            label="Memory"
            value={status.memory_percent}
            unit="%"
            max={100}
            colorClass={gaugeColor(status.memory_percent)}
          />
          <GaugeBar
            label="Disk"
            value={status.disk_percent}
            unit="%"
            max={100}
            colorClass={gaugeColor(status.disk_percent)}
          />
          {status.cpu_temp_celsius !== null && (
            <GaugeBar
              label="CPU Temp"
              value={status.cpu_temp_celsius}
              unit="°C"
              max={100}
              colorClass={tempColor(status.cpu_temp_celsius)}
            />
          )}
          <div className="flex justify-between text-sm pt-1 border-t border-slate-700">
            <span className="text-slate-400">Uptime</span>
            <span className="font-mono font-semibold text-white">
              {formatUptime(status.uptime_seconds)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
