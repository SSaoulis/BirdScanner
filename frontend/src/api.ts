/** Typed fetch wrappers for all /api/* endpoints. */

export interface Detection {
  id: number;
  timestamp: string;
  species: string;
  confidence: number;
  image_path: string;
  thumbnail_path: string;
  track_id: number | null;
  stable_frames: number | null;
  duration_sec: number | null;
  uploaded_at: string | null;
}

export interface SystemStatus {
  cpu_percent: number;
  memory_percent: number;
  disk_percent: number;
  cpu_temp_celsius: number | null;
  uptime_seconds: number;
}

export interface SpeciesSummary {
  species: string;
  count: number;
}

export interface DetectionListParams {
  species?: string;
  from?: string;
  to?: string;
  /** Only return detections with confidence at or above this value (0–1). */
  min_confidence?: number;
  limit?: number;
  offset?: number;
}

async function apiFetch<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`API ${path} → ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  detections: {
    list: (params?: DetectionListParams): Promise<Detection[]> =>
      apiFetch<Detection[]>("/api/detections", params as Record<string, string | number | undefined>),

    get: (id: number): Promise<Detection> =>
      apiFetch<Detection>(`/api/detections/${id}`),
  },

  images: {
    thumbnailUrl: (id: number): string => `/api/images/${id}/thumbnail`,
    fullUrl: (id: number): string => `/api/images/${id}/full`,
    downloadUrl: (ids: number[]): string => `/api/images/download?ids=${ids.join(",")}`,
  },

  system: {
    get: (): Promise<SystemStatus> => apiFetch<SystemStatus>("/api/system"),
  },

  species: {
    list: (): Promise<SpeciesSummary[]> => apiFetch<SpeciesSummary[]>("/api/species"),
  },

  camera: {
    /** URL for an on-demand camera snapshot, proxied from the detector. */
    snapshotUrl: (): string => "/api/camera/snapshot",
  },
};

/** Format a timestamp string as a human-readable "time ago" label. */
export function timeAgo(timestamp: string): string {
  const diff = Date.now() - new Date(timestamp).getTime();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/** Format uptime seconds as "Xd Xh Xm Xs". */
export function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const parts: string[] = [];
  if (d > 0) parts.push(`${d}d`);
  if (h > 0) parts.push(`${h}h`);
  if (m > 0) parts.push(`${m}m`);
  parts.push(`${s}s`);
  return parts.join(" ");
}
