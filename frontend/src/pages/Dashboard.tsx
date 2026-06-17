import { useEffect, useState } from "react";
import { api, type Detection } from "../api";
import { DetectionCard } from "../components/DetectionCard";
import { SystemMonitor } from "../components/SystemMonitor";
import { Lightbox } from "../components/Lightbox";

const RECENT_LIMIT = 10;

export function Dashboard() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Minimum confidence as a 0–100 percentage; 0 means "show all".
  // `sliderConfidence` tracks the live slider position for display only;
  // `minConfidence` is the committed value that drives the fetch and is only
  // updated when the slider is released (so dragging doesn't refetch on every
  // intermediate value).
  const [sliderConfidence, setSliderConfidence] = useState<number>(0);
  const [minConfidence, setMinConfidence] = useState<number>(0);
  // Index into `detections` of the card open in the comparison panel, or null.
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  useEffect(() => {
    setLoading(true);
    api.detections
      .list({
        limit: RECENT_LIMIT,
        min_confidence: minConfidence > 0 ? minConfidence / 100 : undefined,
      })
      .then((data) => {
        setDetections(data);
        setError(null);
        setLoading(false);
        setLightboxIndex(null);
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : "Failed to load detections");
        setLoading(false);
      });
  }, [minConfidence]);

  const currentDetection = lightboxIndex !== null ? detections[lightboxIndex] ?? null : null;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6 space-y-6">
      {/* Comparison panel overlay */}
      {currentDetection && lightboxIndex !== null && (
        <Lightbox
          detection={currentDetection}
          onClose={() => setLightboxIndex(null)}
          onPrev={lightboxIndex > 0 ? () => setLightboxIndex(lightboxIndex - 1) : null}
          onNext={
            lightboxIndex < detections.length - 1
              ? () => setLightboxIndex(lightboxIndex + 1)
              : null
          }
          onDelete={(deletedId) => {
            setDetections((prev) => prev.filter((d) => d.id !== deletedId));
            setLightboxIndex(null);
          }}
        />
      )}

      <header>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
      </header>

      <SystemMonitor />

      <section>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <h2 className="text-lg font-semibold">Recent Detections</h2>

          {/* Minimum confidence slider */}
          <div className="flex items-center gap-3">
            <label
              className="text-xs text-slate-400 font-medium whitespace-nowrap"
              htmlFor="dashboard-confidence"
            >
              Min confidence: {sliderConfidence}%
            </label>
            <input
              id="dashboard-confidence"
              type="range"
              min={0}
              max={100}
              step={1}
              className="accent-emerald-500 w-40"
              value={sliderConfidence}
              onChange={(e) => setSliderConfidence(Number(e.target.value))}
              onMouseUp={() => setMinConfidence(sliderConfidence)}
              onTouchEnd={() => setMinConfidence(sliderConfidence)}
              onKeyUp={() => setMinConfidence(sliderConfidence)}
            />
          </div>
        </div>

        {loading && (
          <p className="text-sm text-slate-500 animate-pulse">Loading…</p>
        )}

        {error && (
          <p className="text-sm text-red-400">{error}</p>
        )}

        {!loading && !error && detections.length === 0 && (
          <p className="text-sm text-slate-500">
            {minConfidence > 0
              ? `No detections at or above ${minConfidence}% confidence.`
              : "No detections yet."}
          </p>
        )}

        {!loading && detections.length > 0 && (
          <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-slate-700">
            {detections.map((d, i) => (
              <DetectionCard
                key={d.id}
                detection={d}
                onOpenLightbox={() => setLightboxIndex(i)}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
