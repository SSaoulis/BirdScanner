import { useEffect, useMemo, useState } from "react";
import { api, ApiError, Settings as SettingsValues, SettingsState } from "../api";

/**
 * Settings page (`/settings`).
 *
 * Loads the detector's runtime settings via `GET /api/settings`, lets the user
 * edit them in grouped sections, and saves changed fields via `POST
 * /api/settings`. Live-safe fields (detection/classification thresholds, ignore
 * list, debug logging) take effect immediately; restart-only fields are badged
 * and, once changed + saved, surface an "Apply & restart detector" banner that
 * relaunches the detector so they take effect.
 *
 * All writes are proxied by the API to the detector's control server (the API
 * mounts the data volume read-only and cannot change settings itself).
 */
export function Settings() {
  const [state, setState] = useState<SettingsState | null>(null);
  const [form, setForm] = useState<SettingsValues | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [restarting, setRestarting] = useState(false);

  async function load() {
    setLoadError(null);
    try {
      const data = await api.settings.get();
      setState(data);
      setForm(data.settings);
    } catch (e) {
      setLoadError(
        e instanceof ApiError && e.status === 503
          ? "The detector is offline, so settings can't be loaded right now."
          : e instanceof Error
            ? e.message
            : "Failed to load settings",
      );
    }
  }

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const restartFields = useMemo(
    () => new Set(state?.restart_fields ?? []),
    [state],
  );

  // Which fields differ from the last-loaded server values (what we'll save).
  const changed = useMemo<(keyof SettingsValues)[]>(() => {
    if (!state || !form) return [];
    return (Object.keys(form) as (keyof SettingsValues)[]).filter(
      (k) => JSON.stringify(form[k]) !== JSON.stringify(state.settings[k]),
    );
  }, [state, form]);

  const dirty = changed.length > 0;

  function setField<K extends keyof SettingsValues>(key: K, value: SettingsValues[K]) {
    setForm((prev) => (prev ? { ...prev, [key]: value } : prev));
    setSaved(false);
    setSaveError(null);
  }

  async function save() {
    if (!form || !dirty) return;
    setSaving(true);
    setSaveError(null);
    setSaved(false);
    const updates: Partial<SettingsValues> = {};
    for (const key of changed) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (updates as any)[key] = form[key];
    }
    try {
      const next = await api.settings.update(updates);
      setState(next);
      setForm(next.settings);
      setSaved(true);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Failed to save settings");
    } finally {
      setSaving(false);
    }
  }

  function revert() {
    if (state) setForm(state.settings);
    setSaveError(null);
    setSaved(false);
  }

  async function restart() {
    setRestarting(true);
    setSaveError(null);
    try {
      await api.settings.restart();
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Failed to restart detector");
      setRestarting(false);
    }
  }

  if (loadError) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-8 space-y-4">
        <PageHeader />
        <div className="rounded-lg border border-rust/40 bg-rust/10 px-4 py-3 text-sm text-rust">
          {loadError}
        </div>
        <button onClick={() => void load()} className={secondaryButton}>
          Try again
        </button>
      </div>
    );
  }

  if (!form || !state) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-8 space-y-4">
        <PageHeader />
        <p className="text-sm text-bark">Loading settings…</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-8 space-y-6">
      <PageHeader />

      {state.needs_restart && (
        <div className="rounded-xl border border-gold/50 bg-gold/10 px-4 py-3 text-sm text-ink">
          <p className="font-semibold">A restart is needed to finish applying your changes.</p>
          <p className="mt-1 text-bark">
            Some settings only take effect when the detector restarts. This briefly
            interrupts detection while it relaunches.
          </p>
          <button
            onClick={() => void restart()}
            disabled={restarting}
            className={`${primaryButton} mt-3`}
          >
            {restarting ? "Restarting…" : "Apply & restart detector"}
          </button>
        </div>
      )}

      <Section title="Detection" eyebrow="What counts as a sighting">
        <SliderField
          label="Detection confidence"
          help="How sure the camera must be that it sees an object before it's tracked."
          value={form.detection_threshold}
          onChange={(v) => setField("detection_threshold", v)}
          restart={restartFields.has("detection_threshold")}
        />
        <NumberField
          label="Stability duration"
          suffix="sec"
          step={0.1}
          min={0}
          help="How long a bird must stay in view before it's classified."
          value={form.stability_seconds}
          onChange={(v) => setField("stability_seconds", v)}
          restart={restartFields.has("stability_seconds")}
        />
        <ChipsField
          label="Ignore these object types"
          placeholder="Add object type…"
          help="The camera detects all sorts of objects, not just birds. List the ones to drop before tracking so their false positives don't fill the logs (e.g. “bench”). Case-insensitive."
          value={form.excluded_classes}
          onChange={(v) => setField("excluded_classes", v)}
          restart={restartFields.has("excluded_classes")}
        />
      </Section>

      <Section title="Saving" eyebrow="What gets kept">
        <SliderField
          label="Classification confidence to save"
          help="Only save a sighting when the species classifier is at least this sure. Raise it to cut noisy gallery entries."
          value={form.classification_threshold}
          onChange={(v) => setField("classification_threshold", v)}
          restart={restartFields.has("classification_threshold")}
        />
        <ChipsField
          label="Ignore these species"
          help="Species listed here are never saved, even when classified (e.g. “Unknown”). Case-insensitive."
          value={form.ignore_species}
          onChange={(v) => setField("ignore_species", v)}
          restart={restartFields.has("ignore_species")}
        />
        <TextField
          label="Save location"
          help="Folder where images and clips are written. Must stay on the shared data volume so the gallery can find them."
          value={form.image_dir}
          onChange={(v) => setField("image_dir", v)}
          restart={restartFields.has("image_dir")}
        />
      </Section>

      <Section title="Video clips" eyebrow="Short recordings per sighting">
        <ToggleField
          label="Save a clip per sighting"
          help="Records a short mp4 around each saved detection, alongside the still."
          value={form.video_save}
          onChange={(v) => setField("video_save", v)}
          restart={restartFields.has("video_save")}
        />
        <NumberField
          label="Pre-roll"
          suffix="sec"
          step={0.5}
          min={0}
          help="Seconds of footage kept from before the sighting."
          value={form.video_pre_roll_seconds}
          onChange={(v) => setField("video_pre_roll_seconds", v)}
          restart={restartFields.has("video_pre_roll_seconds")}
        />
        <NumberField
          label="Post-roll"
          suffix="sec"
          step={0.5}
          min={0}
          help="Seconds of footage recorded after the sighting triggers."
          value={form.video_post_roll_seconds}
          onChange={(v) => setField("video_post_roll_seconds", v)}
          restart={restartFields.has("video_post_roll_seconds")}
        />
      </Section>

      <Section title="System" eyebrow="How the detector runs">
        <ToggleField
          label="Multithreaded classification"
          help="Run classification on a background thread so the camera never stalls."
          value={form.multithread}
          onChange={(v) => setField("multithread", v)}
          restart={restartFields.has("multithread")}
        />
        <ToggleField
          label="Debug logging"
          help="Verbose track-lifecycle logs. Handy when diagnosing, noisy otherwise."
          value={form.debug}
          onChange={(v) => setField("debug", v)}
          restart={restartFields.has("debug")}
        />
      </Section>

      <Section title="Location" eyebrow="Where the camera is">
        <CoordinateField
          label="Latitude"
          min={-90}
          max={90}
          help="Deployment latitude in degrees (−90 to 90). Used with longitude to build the geomodel's location prior. Leave blank to disable the prior."
          value={form.latitude}
          onChange={(v) => setField("latitude", v)}
          restart={restartFields.has("latitude")}
        />
        <CoordinateField
          label="Longitude"
          min={-180}
          max={180}
          help="Deployment longitude in degrees (−180 to 180). Leave blank to disable the geomodel's location prior."
          value={form.longitude}
          onChange={(v) => setField("longitude", v)}
          restart={restartFields.has("longitude")}
        />
      </Section>

      {/* Action bar */}
      <div className="sticky bottom-0 -mx-6 border-t border-line bg-paper/90 px-6 py-4 backdrop-blur">
        <div className="flex items-center justify-between gap-4">
          <p className="text-sm text-bark">
            {saveError ? (
              <span className="text-rust">{saveError}</span>
            ) : saved ? (
              "Saved. Live settings applied immediately."
            ) : dirty ? (
              `${changed.length} unsaved change${changed.length === 1 ? "" : "s"}.`
            ) : (
              "All changes saved."
            )}
          </p>
          <div className="flex gap-2">
            <button onClick={revert} disabled={!dirty || saving} className={secondaryButton}>
              Revert
            </button>
            <button onClick={() => void save()} disabled={!dirty || saving} className={primaryButton}>
              {saving ? "Saving…" : "Save changes"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const primaryButton =
  "rounded-lg bg-gold px-4 py-2 text-sm font-semibold text-card shadow-sm transition-colors hover:bg-gold-deep disabled:cursor-not-allowed disabled:opacity-50";
const secondaryButton =
  "rounded-lg border border-line bg-card px-4 py-2 text-sm font-semibold text-bark transition-colors hover:text-ink disabled:cursor-not-allowed disabled:opacity-50";

function PageHeader() {
  return (
    <header>
      <p className="eyebrow mb-2">The controls</p>
      <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">Settings</h1>
      <p className="mt-2 text-sm text-bark">
        Tune what the detector looks for, what it keeps, and how it runs.
      </p>
    </header>
  );
}

function Section({
  title,
  eyebrow,
  children,
}: {
  title: string;
  eyebrow: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-line bg-card p-6 shadow-plate space-y-5">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h2 className="font-display text-xl font-semibold text-ink">{title}</h2>
      </div>
      <div className="space-y-5">{children}</div>
    </section>
  );
}

/** A field wrapper: label + optional restart badge + help text, then control. */
function Field({
  label,
  help,
  restart,
  children,
}: {
  label: string;
  help: string;
  restart?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between gap-3">
        <label className="text-sm font-semibold text-ink">{label}</label>
        {restart && <RestartBadge />}
      </div>
      {children}
      <p className="text-xs text-bark">{help}</p>
    </div>
  );
}

function RestartBadge() {
  return (
    <span
      title="Takes effect after the detector restarts"
      className="rounded-full border border-gold/50 bg-gold/10 px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-gold-deep"
    >
      Restart
    </span>
  );
}

function SliderField({
  label,
  help,
  value,
  onChange,
  restart,
}: {
  label: string;
  help: string;
  value: number;
  onChange: (v: number) => void;
  restart?: boolean;
}) {
  return (
    <Field label={label} help={help} restart={restart}>
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(value * 100)}
          onChange={(e) => onChange(Number(e.target.value) / 100)}
          className="h-2 flex-1 cursor-pointer accent-gold"
        />
        <span className="tnum w-12 text-right text-sm text-ink">
          {Math.round(value * 100)}%
        </span>
      </div>
    </Field>
  );
}

function NumberField({
  label,
  help,
  value,
  onChange,
  restart,
  suffix,
  step,
  min,
}: {
  label: string;
  help: string;
  value: number;
  onChange: (v: number) => void;
  restart?: boolean;
  suffix?: string;
  step?: number;
  min?: number;
}) {
  return (
    <Field label={label} help={help} restart={restart}>
      <div className="flex items-center gap-2">
        <input
          type="number"
          value={value}
          step={step ?? 1}
          min={min}
          onChange={(e) => onChange(Number(e.target.value))}
          className="tnum w-28 rounded-lg border border-line bg-paper px-3 py-1.5 text-sm text-ink focus-visible:outline-none"
        />
        {suffix && <span className="text-sm text-bark">{suffix}</span>}
      </div>
    </Field>
  );
}

/**
 * An optional coordinate input (latitude/longitude).
 *
 * Renders a number input bounded to `[min, max]` degrees. An empty input maps
 * to `null` (the coordinate is unset, so the geomodel prior is not built), and a
 * `null` value renders as a blank field with a "Not set" placeholder.
 */
function CoordinateField({
  label,
  help,
  value,
  onChange,
  restart,
  min,
  max,
}: {
  label: string;
  help: string;
  value: number | null;
  onChange: (v: number | null) => void;
  restart?: boolean;
  min: number;
  max: number;
}) {
  return (
    <Field label={label} help={help} restart={restart}>
      <div className="flex items-center gap-2">
        <input
          type="number"
          value={value ?? ""}
          step="any"
          min={min}
          max={max}
          placeholder="Not set"
          onChange={(e) => {
            const text = e.target.value.trim();
            onChange(text === "" ? null : Number(text));
          }}
          className="tnum w-40 rounded-lg border border-line bg-paper px-3 py-1.5 text-sm text-ink focus-visible:outline-none"
        />
        <span className="text-sm text-bark">°</span>
      </div>
    </Field>
  );
}

function TextField({
  label,
  help,
  value,
  onChange,
  restart,
}: {
  label: string;
  help: string;
  value: string;
  onChange: (v: string) => void;
  restart?: boolean;
}) {
  return (
    <Field label={label} help={help} restart={restart}>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-line bg-paper px-3 py-1.5 text-sm text-ink focus-visible:outline-none"
      />
    </Field>
  );
}

function ToggleField({
  label,
  help,
  value,
  onChange,
  restart,
}: {
  label: string;
  help: string;
  value: boolean;
  onChange: (v: boolean) => void;
  restart?: boolean;
}) {
  return (
    <Field label={label} help={help} restart={restart}>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        onClick={() => onChange(!value)}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
          value ? "bg-gold" : "bg-line"
        }`}
      >
        <span
          className={`inline-block h-5 w-5 transform rounded-full bg-card shadow-sm transition-transform ${
            value ? "translate-x-5" : "translate-x-0.5"
          }`}
        />
      </button>
    </Field>
  );
}

function ChipsField({
  label,
  help,
  value,
  onChange,
  restart,
  placeholder = "Add species…",
}: {
  label: string;
  help: string;
  value: string[];
  onChange: (v: string[]) => void;
  restart?: boolean;
  placeholder?: string;
}) {
  const [draft, setDraft] = useState("");

  function add() {
    const name = draft.trim();
    if (name && !value.some((v) => v.toLowerCase() === name.toLowerCase())) {
      onChange([...value, name]);
    }
    setDraft("");
  }

  return (
    <Field label={label} help={help} restart={restart}>
      <div className="flex flex-wrap items-center gap-2">
        {value.map((name) => (
          <span
            key={name}
            className="inline-flex items-center gap-1 rounded-full border border-line bg-paper px-2.5 py-1 text-xs text-ink"
          >
            {name}
            <button
              type="button"
              aria-label={`Remove ${name}`}
              onClick={() => onChange(value.filter((v) => v !== name))}
              className="text-bark hover:text-rust"
            >
              ×
            </button>
          </span>
        ))}
        <input
          type="text"
          value={draft}
          placeholder={placeholder}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              add();
            }
          }}
          onBlur={add}
          className="min-w-[8rem] flex-1 rounded-lg border border-line bg-paper px-3 py-1.5 text-sm text-ink focus-visible:outline-none"
        />
      </div>
    </Field>
  );
}
