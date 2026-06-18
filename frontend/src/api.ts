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
  /** Detection box left edge as a fraction [0, 1] of image width (null for legacy rows). */
  box_x: number | null;
  /** Detection box top edge as a fraction [0, 1] of image height (null for legacy rows). */
  box_y: number | null;
  /** Detection box width as a fraction [0, 1] of image width (null for legacy rows). */
  box_w: number | null;
  /** Detection box height as a fraction [0, 1] of image height (null for legacy rows). */
  box_h: number | null;
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

export interface SpeciesReferenceImage {
  /** Ready-to-use API path for the reference image; render directly in <img src>. */
  url: string;
  attribution: string;
  license: string | null;
}

export interface SpeciesReference {
  common_name: string;
  scientific_name: string | null;
  summary: string;
  behaviour: string | null;
  wikipedia_url: string | null;
  images: SpeciesReferenceImage[];
}

/** Error thrown by apiFetch carrying the HTTP status code, so callers can branch on 404. */
export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
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

/** A normalized crop box (fractions in [0, 1]) over the displayed preview. */
export interface NormalizedBox {
  nx: number;
  ny: number;
  nw: number;
  nh: number;
}

/** The detection crop region as reported by the detector. */
export interface CropState {
  /** Crop rectangle in raw sensor pixels. */
  x: number;
  y: number;
  w: number;
  h: number;
  /** Same region as a normalized box over the displayed (flipped) preview. */
  norm: NormalizedBox;
  /** Full sensor dimensions, for rendering the preview at the true aspect. */
  sensor_w: number;
  sensor_h: number;
}

async function apiFetch<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new ApiError(res.status, `API ${path} → ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${path} → ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

async function apiDelete(path: string): Promise<void> {
  const res = await fetch(path, { method: "DELETE" });
  if (!res.ok) throw new Error(`API ${path} → ${res.status} ${res.statusText}`);
}

export const api = {
  detections: {
    list: (params?: DetectionListParams): Promise<Detection[]> =>
      apiFetch<Detection[]>("/api/detections", params as Record<string, string | number | undefined>),

    get: (id: number): Promise<Detection> =>
      apiFetch<Detection>(`/api/detections/${id}`),

    /** Permanently delete a detection (its DB row + image files). */
    delete: (id: number): Promise<void> =>
      apiDelete(`/api/detections/${id}`),
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

    /**
     * Fetch reference data (images + species info) for a species by its
     * common name. Rejects with an ApiError (status 404) when no reference
     * data exists for the species.
     */
    reference: (name: string): Promise<SpeciesReference> =>
      apiFetch<SpeciesReference>(`/api/species/${encodeURIComponent(name)}/reference`),
  },

  camera: {
    /** URL for an on-demand snapshot of the current (cropped) feed. */
    snapshotUrl: (): string => "/api/camera/snapshot",

    /** URL for a full-sensor snapshot used by the crop editor. */
    fullSnapshotUrl: (): string => "/api/camera/snapshot/full",

    /** Fetch the detector's current detection-crop region. */
    getCrop: (): Promise<CropState> => apiFetch<CropState>("/api/camera/crop"),

    /** Apply a new crop region (a normalized box, or `{ reset: true }`). */
    setCrop: (body: NormalizedBox | { reset: true }): Promise<CropState> =>
      postJson<CropState>("/api/camera/crop", body),
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
