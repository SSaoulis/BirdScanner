import { useEffect, useState } from "react";
import { api, type Detection } from "../api";
import { DetectionCard } from "../components/DetectionCard";
import { SystemMonitor } from "../components/SystemMonitor";

const RECENT_LIMIT = 10;

export function Dashboard() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.detections
      .list({ limit: RECENT_LIMIT })
      .then((data) => {
        setDetections(data);
        setLoading(false);
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : "Failed to load detections");
        setLoading(false);
      });
  }, []);

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6 space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
      </header>

      <SystemMonitor />

      <section>
        <h2 className="text-lg font-semibold mb-3">Recent Detections</h2>

        {loading && (
          <p className="text-sm text-slate-500 animate-pulse">Loading…</p>
        )}

        {error && (
          <p className="text-sm text-red-400">{error}</p>
        )}

        {!loading && !error && detections.length === 0 && (
          <p className="text-sm text-slate-500">No detections yet.</p>
        )}

        {!loading && detections.length > 0 && (
          <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-700">
            {detections.map((d) => (
              <DetectionCard key={d.id} detection={d} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
