/** Typed fetch wrappers for all /api/* endpoints. */

export interface Detection {
  id: number;
  timestamp: string;
  species: string;
  /** Species-classification confidence (ConvNeXt) in [0, 1]. */
  confidence: number;
  /** Object-detection confidence (YOLO11n) in [0, 1]; null for legacy rows. */
  detection_confidence: number | null;
  image_path: string;
  thumbnail_path: string;
  /** Path to the saved mp4 clip; null for legacy rows, disabled video, or while the clip is still encoding. */
  video_path: string | null;
  /**
   * Why this sighting has no clip, when `video_path` is null:
   * `"recorder_busy"` (a clip for another sighting was recording — only one
   * records at a time), `"disabled"` (video recording is off), or null for
   * legacy rows written before the reason was persisted.
   */
  no_video_reason: string | null;
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
  /** True when a user manually overrode the classifier's species; null/false otherwise. */
  corrected: boolean | null;
  /**
   * The classifier's original top-1 species, preserved when a user corrects the
   * detection so the model's guess stays on record. `confidence` is that guess's
   * score. Null when the detection was never corrected.
   */
  original_species: string | null;
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

/** One network-throughput sample from the passive NIC monitor. */
export interface NetworkSample {
  /** Unix timestamp (seconds) the sample was taken. */
  t: number;
  /** Download rate at that instant, in kilobits/sec. */
  rx_kbps: number;
  /** Upload rate at that instant, in kilobits/sec. */
  tx_kbps: number;
}

/** A window of network throughput samples. */
export interface NetworkHistory {
  /** Nominal seconds between samples (the sampler cadence). */
  interval_sec: number;
  /** Samples within the requested window, oldest first. */
  samples: NetworkSample[];
}

/** Selectable history windows for the usage graph. */
export type NetworkRange = "5m" | "30m" | "1h";

/** Result of one on-demand internet speed test. */
export interface SpeedTestResult {
  download_mbps: number;
  upload_mbps: number;
  download_bytes: number;
  upload_bytes: number;
  /** Unix timestamp (seconds) the test completed. */
  ran_at: number;
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

/** The user-editable detector settings (see birdscanner/detector/settings.py). */
export interface Settings {
  /** Minimum YOLO object-detection confidence to keep a detection (0–1). */
  detection_threshold: number;
  /** Minimum species-classification confidence before a detection is saved (0–1). */
  classification_threshold: number;
  /** Species never saved even when classified (matched case-insensitively). */
  ignore_species: string[];
  /** Seconds a track must be stable before classification fires. */
  stability_seconds: number;
  /** Root directory saved images/clips are written to. */
  image_dir: string;
  /** Whether to save a short mp4 clip per detection. */
  video_save: boolean;
  /** Seconds of buffered footage prepended to a clip. */
  video_pre_roll_seconds: number;
  /** Seconds recorded after a clip triggers. */
  video_post_roll_seconds: number;
  /** Run classification on a background thread. */
  multithread: boolean;
  /** Enable DEBUG-level tracking logs. */
  debug: boolean;
  /** Deployment latitude in degrees for the geomodel prior; null when unset. */
  latitude: number | null;
  /** Deployment longitude in degrees for the geomodel prior; null when unset. */
  longitude: number | null;
}

/** Settings plus the metadata the UI needs to badge/handle restart-only fields. */
export interface SettingsState {
  settings: Settings;
  /** True when a restart-only field changed since the detector last booted. */
  needs_restart: boolean;
  /** Field names that only take effect after a detector restart. */
  restart_fields: string[];
  /** Field names applied live (take effect immediately). */
  live_fields: string[];
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
  if (!res.ok) {
    // Surface the API's error detail (e.g. a validation message) when present,
    // so callers can show why a settings update was rejected. FastAPI wraps it
    // as {"detail": "..."}.
    const raw = await res.text().catch(() => "");
    let message = raw;
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.detail === "string") message = parsed.detail;
    } catch {
      /* not JSON — keep the raw body */
    }
    throw new ApiError(res.status, message || `API ${path} → ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

async function patchJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    // Surface the API's error detail (e.g. "Unknown species 'X'") when present.
    // FastAPI wraps it as {"detail": "..."}.
    const raw = await res.text().catch(() => "");
    let message = raw;
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.detail === "string") message = parsed.detail;
    } catch {
      /* not JSON — keep the raw body */
    }
    throw new ApiError(res.status, message || `API ${path} → ${res.status} ${res.statusText}`);
  }
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

    /**
     * Correct a detection's species. Returns the updated detection (with
     * `corrected` set and image paths moved to the new species folder). Rejects
     * with an ApiError(400) carrying the message for an unknown species, or 503
     * when the detector is unreachable.
     */
    correct: (id: number, species: string): Promise<Detection> =>
      patchJson<Detection>(`/api/detections/${id}`, { species }),
  },

  images: {
    thumbnailUrl: (id: number): string => `/api/images/${id}/thumbnail`,
    fullUrl: (id: number): string => `/api/images/${id}/full`,
    videoUrl: (id: number): string => `/api/images/${id}/video`,
    downloadUrl: (ids: number[]): string => `/api/images/download?ids=${ids.join(",")}`,
  },

  system: {
    get: (): Promise<SystemStatus> => apiFetch<SystemStatus>("/api/system"),
  },

  network: {
    /** Fetch passive NIC throughput samples for the given time window. */
    history: (range: NetworkRange): Promise<NetworkHistory> =>
      apiFetch<NetworkHistory>("/api/network/history", { range }),

    /** Run an on-demand internet speed test (download + upload). */
    speedTest: (): Promise<SpeedTestResult> =>
      postJson<SpeedTestResult>("/api/network/speedtest", {}),
  },

  species: {
    list: (): Promise<SpeciesSummary[]> => apiFetch<SpeciesSummary[]>("/api/species"),

    /**
     * Fetch the classifier's full species vocabulary (every label the model can
     * predict), used to populate the correction picker. Rejects with an
     * ApiError(503) when the detector is unreachable.
     */
    vocabulary: (): Promise<string[]> => apiFetch<string[]>("/api/species/vocabulary"),

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

  settings: {
    /** Fetch the detector's current settings + restart metadata. */
    get: (): Promise<SettingsState> => apiFetch<SettingsState>("/api/settings"),

    /**
     * Apply a partial settings update. Rejects with an ApiError (status 400)
     * carrying the validation message when a value is invalid, or 503 when the
     * detector is unreachable.
     */
    update: (updates: Partial<Settings>): Promise<SettingsState> =>
      postJson<SettingsState>("/api/settings", updates),

    /** Ask the detector to restart so restart-only settings take effect. */
    restart: (): Promise<{ status: string }> =>
      postJson<{ status: string }>("/api/settings/restart", {}),
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
