import { type ReactNode } from "react";
import { type Detection } from "../api";

/** One `[species, score]` pair from a persisted top-k JSON array. */
type ScorePair = [string, number];

/**
 * Parse a persisted top-k JSON string (`classifier_scores` / `geo_scores`) into
 * `[species, score]` pairs, defensively. Returns null for legacy rows (null
 * string), malformed JSON, or an empty/invalid array so callers can fall back.
 */
function parseScores(json: string | null): ScorePair[] | null {
  if (!json) return null;
  try {
    const parsed = JSON.parse(json);
    if (!Array.isArray(parsed)) return null;
    const pairs = parsed.filter(
      (p): p is ScorePair =>
        Array.isArray(p) &&
        p.length === 2 &&
        typeof p[0] === "string" &&
        typeof p[1] === "number",
    );
    return pairs.length > 0 ? pairs : null;
  } catch {
    return null;
  }
}

/** Format a raw geomodel weight (a small `p(y|x)·p(y|c)` product) for display. */
function formatWeight(score: number): string {
  if (score >= 0.01) return score.toFixed(3);
  return score.toExponential(1);
}

interface ScoreBarProps {
  /** Row label (usually a species name). */
  label: string;
  /** Bar fill as a fraction in [0, 1] of the row's own scale. */
  barFrac: number;
  /** Right-aligned numeric readout (already formatted). */
  valueText: string;
  /** Highlight the winning row in gold (others render in sage). */
  highlight?: boolean;
}

/** A single labelled horizontal bar used by the score lists. */
function ScoreBar({ label, barFrac, valueText, highlight }: ScoreBarProps) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={`min-w-0 truncate text-sm ${
            highlight ? "font-display text-ink" : "text-bark"
          }`}
        >
          {label}
        </span>
        <span className="tnum shrink-0 text-xs text-bark">{valueText}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-line/60">
        <div
          className={`h-full rounded-full ${highlight ? "bg-gold" : "bg-sage"}`}
          style={{ width: `${Math.max(2, Math.min(1, barFrac) * 100)}%` }}
        />
      </div>
    </div>
  );
}

interface StatSectionProps {
  /** Small-caps eyebrow heading for the section. */
  title: string;
  children: ReactNode;
}

/** A titled block within the stats panel. */
function StatSection({ title, children }: StatSectionProps) {
  return (
    <section className="flex flex-col gap-2">
      <p className="eyebrow">{title}</p>
      {children}
    </section>
  );
}

/**
 * The left-hand "Advanced stats" panel content for the Lightbox: the per-prediction
 * telemetry we persist — the classifier's top-k distribution, the geomodel's
 * reweighting, the object-detection score, the stable-frame count, and record
 * metadata. Each section is hidden when its source data is null (legacy rows, or no
 * geomodel location configured), so old detections degrade gracefully. Purely
 * presentational — it reads everything off the passed `detection`.
 */
export function AdvancedStatsPane({ detection }: { detection: Detection }) {
  const {
    species,
    confidence,
    detection_confidence: detConf,
    stable_frames: stableFrames,
    track_id: trackId,
    classifier_species: clfSpecies,
    classifier_confidence: clfConf,
    original_species: originalSpecies,
    timestamp,
  } = detection;
  const corrected = detection.corrected === true;
  const classifierScores = parseScores(detection.classifier_scores);
  const geoScores = parseScores(detection.geo_scores);
  // The geomodel update ran iff it recorded the classifier's pre-adjustment pick.
  const geoRan = clfSpecies != null;
  const confidencePct = (confidence * 100).toFixed(1);

  return (
    <div className="flex flex-col gap-5">
      {/* ── Prediction (headline) ── */}
      <StatSection title="Prediction">
        <p className="font-display text-xl font-medium text-ink">{species}</p>
        {corrected ? (
          <div className="flex flex-col gap-0.5">
            <span className="font-display text-sm italic text-gold-deep">
              ✎ Set by you
            </span>
            {originalSpecies && (
              <span className="text-xs text-bark">
                model saw <span className="tnum">{confidencePct}%</span>{" "}
                {originalSpecies}
              </span>
            )}
          </div>
        ) : (
          <ScoreBar
            label="Confidence"
            barFrac={confidence}
            valueText={`${confidencePct}%`}
            highlight
          />
        )}
      </StatSection>

      {/* ── Top predictions (raw classifier softmax) ── */}
      <StatSection title="Top predictions">
        {classifierScores ? (
          <div className="flex flex-col gap-2">
            {classifierScores.map(([name, prob], i) => (
              <ScoreBar
                key={`${name}-${i}`}
                label={name}
                barFrac={prob}
                valueText={`${(prob * 100).toFixed(1)}%`}
                highlight={i === 0}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-bark">
            Only the top prediction was recorded for this sighting.
          </p>
        )}
      </StatSection>

      {/* ── Location prior (geomodel) — only when the update ran ── */}
      {geoRan && (
        <StatSection title="Location prior (geomodel)">
          {clfSpecies === species ? (
            <p className="text-sm text-ink/85">
              Kept <span className="font-display">{species}</span> · confidence{" "}
              <span className="tnum">
                {clfConf != null ? (clfConf * 100).toFixed(1) : "—"}%
              </span>{" "}
              → <span className="tnum text-gold-deep">{confidencePct}%</span>
            </p>
          ) : (
            <p className="flex flex-wrap items-baseline gap-1.5 text-sm text-ink/85">
              <span className="font-display">{clfSpecies}</span>
              <span className="tnum text-bark">
                {clfConf != null ? (clfConf * 100).toFixed(1) : "—"}%
              </span>
              <span aria-hidden="true" className="text-bark">
                →
              </span>
              <span className="font-display text-gold-deep">{species}</span>
              <span className="tnum text-gold-deep">{confidencePct}%</span>
            </p>
          )}
          {geoScores && (
            <div className="flex flex-col gap-2">
              <p className="text-xs text-bark">
                Geomodel-weighted scores (relative)
              </p>
              {geoScores.map(([name, score], i) => (
                <ScoreBar
                  key={`${name}-${i}`}
                  label={name}
                  barFrac={score / geoScores[0][1]}
                  valueText={formatWeight(score)}
                  highlight={name === species}
                />
              ))}
            </div>
          )}
        </StatSection>
      )}

      {/* ── Detection quality ── */}
      {(detConf != null || stableFrames != null) && (
        <StatSection title="Detection quality">
          {detConf != null && (
            <ScoreBar
              label="Object detection (YOLO)"
              barFrac={detConf}
              valueText={`${(detConf * 100).toFixed(1)}%`}
            />
          )}
          {stableFrames != null && (
            <div className="flex items-baseline justify-between text-sm">
              <span className="text-bark">Stable frames tracked</span>
              <span className="tnum text-ink">{stableFrames}</span>
            </div>
          )}
        </StatSection>
      )}

      {/* ── Record details (muted footer) ── */}
      <StatSection title="Record">
        <dl className="flex flex-col gap-1 text-xs text-bark">
          {trackId != null && (
            <div className="flex justify-between gap-2">
              <dt>Track ID</dt>
              <dd className="tnum text-ink">{trackId}</dd>
            </div>
          )}
          <div className="flex justify-between gap-2">
            <dt>Captured</dt>
            <dd className="tnum text-ink">
              {new Date(timestamp).toLocaleString()}
            </dd>
          </div>
          {corrected && originalSpecies && (
            <div className="flex justify-between gap-2">
              <dt>Correction</dt>
              <dd className="text-ink">model saw {originalSpecies}</dd>
            </div>
          )}
          <div className="flex justify-between gap-2">
            <dt>Clip</dt>
            <dd className="text-ink">
              {detection.video_path != null
                ? "saved"
                : detection.no_video_reason === "recorder_busy"
                  ? "recorder was busy"
                  : detection.no_video_reason === "disabled"
                    ? "recording off"
                    : "none"}
            </dd>
          </div>
        </dl>
      </StatSection>
    </div>
  );
}
