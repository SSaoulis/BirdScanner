import { api, timeAgo, type Detection } from "../api";

interface DetectionCardProps {
  detection: Detection;
}

export function DetectionCard({ detection }: DetectionCardProps) {
  const { id, species, confidence, timestamp } = detection;
  const thumbnailUrl = api.images.thumbnailUrl(id);
  const confidencePct = (confidence * 100).toFixed(1);

  return (
    <a
      href={api.images.fullUrl(id)}
      target="_blank"
      rel="noopener noreferrer"
      className="flex-shrink-0 w-44 bg-slate-800 rounded-xl overflow-hidden hover:ring-2 hover:ring-emerald-500 transition-all duration-150 group"
    >
      <div className="relative w-full h-36 bg-slate-900">
        <img
          src={thumbnailUrl}
          alt={species}
          className="w-full h-full object-cover group-hover:opacity-90 transition-opacity duration-150"
          loading="lazy"
        />
      </div>
      <div className="p-2 space-y-0.5">
        <p className="text-sm font-semibold text-white truncate" title={species}>
          {species}
        </p>
        <div className="flex items-center justify-between">
          <span className="text-xs text-emerald-400 font-mono">{confidencePct}%</span>
          <span className="text-xs text-slate-500">{timeAgo(timestamp)}</span>
        </div>
      </div>
    </a>
  );
}
