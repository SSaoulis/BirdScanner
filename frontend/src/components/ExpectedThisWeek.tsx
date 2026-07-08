import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type ExpectedSpecies, type ExpectedThisWeek as ExpectedData } from "../api";

/**
 * Small square thumbnail for an expected species, pulled from the offline
 * reference bank. When the bank has no image for the species (or isn't
 * deployed) the request 404s and we fall back to a monogram tile, so the row
 * layout never shifts.
 */
function SpeciesThumb({ species }: { species: string }) {
  const [failed, setFailed] = useState(false);
  const monogram = species.trim().charAt(0).toUpperCase() || "·";

  if (failed) {
    return (
      <div
        className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-md bg-sage/20 font-display text-base text-sage-deep"
        aria-hidden="true"
      >
        {monogram}
      </div>
    );
  }
  return (
    <img
      src={api.species.referenceImageUrl(species)}
      alt=""
      className="h-10 w-10 flex-shrink-0 rounded-md object-cover shadow-plate"
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

/**
 * One field-guide index row: thumbnail, common name in the journal serif, and a
 * thin "occurrence" bar. The bar is sized relative to the most-likely species
 * in the set so the ranking reads at a glance; no percentage is shown because
 * the prior is an occurrence likelihood, not a chance-of-seeing-today.
 */
function ExpectedRow({ item, max }: { item: ExpectedSpecies; max: number }) {
  const fraction = max > 0 ? Math.max(item.score / max, 0.06) : 0;
  return (
    <div className="flex items-center gap-3">
      <SpeciesThumb species={item.species} />
      <span className="min-w-0 flex-1 truncate font-display text-[0.95rem] text-ink" title={item.species}>
        {item.species}
      </span>
      <span
        className="h-1.5 w-16 flex-shrink-0 overflow-hidden rounded-full bg-line"
        role="img"
        aria-label={`Occurrence relative to the most likely species: ${(fraction * 100).toFixed(0)}%`}
      >
        <span
          className="block h-full rounded-full bg-sage-deep"
          style={{ width: `${fraction * 100}%` }}
        />
      </span>
    </div>
  );
}

/**
 * "In season · this week" — a quiet field-journal band listing the species the
 * geomodel expects near the feeder for the current week. Fetches its own data
 * on mount (independent of the Dashboard's other controls).
 *
 * States:
 * - loading: placeholder rows, no layout shift.
 * - no location configured: an invitation to set one in Settings.
 * - priors built: the ranked list.
 * - empty (priors built but no rows) or error: renders nothing, so the page
 *   stays clean rather than showing a broken panel.
 */
export function ExpectedThisWeek() {
  const [data, setData] = useState<ExpectedData | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api.species
      .expected()
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setFailed(true);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // A failed fetch is not worth a visible error on the Today page — just omit
  // the panel.
  if (failed) return null;

  const Shell = ({ children }: { children: React.ReactNode }) => (
    <section className="rounded-xl border border-line bg-card p-5 shadow-plate">
      <p className="eyebrow mb-3">In season · this week</p>
      {children}
    </section>
  );

  if (loading) {
    return (
      <Shell>
        <div className="grid animate-pulse grid-cols-1 gap-3 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="h-10 w-10 flex-shrink-0 rounded-md bg-line/60" />
              <div className="h-3 flex-1 rounded bg-line/60" />
              <div className="h-1.5 w-16 flex-shrink-0 rounded-full bg-line/60" />
            </div>
          ))}
        </div>
      </Shell>
    );
  }

  if (!data) return null;

  // No location set yet: nudge the user toward Settings so the prior can build.
  if (data.latitude === null && data.species.length === 0) {
    return (
      <Shell>
        <p className="text-sm text-bark">
          Add your location in{" "}
          <Link to="/settings" className="font-medium text-gold-deep underline underline-offset-2">
            Settings
          </Link>{" "}
          to see which species to expect this time of year.
        </p>
      </Shell>
    );
  }

  if (data.species.length === 0) return null;

  const max = data.species[0].score;
  return (
    <Shell>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 sm:gap-x-8">
        {data.species.map((item) => (
          <ExpectedRow key={item.species} item={item} max={max} />
        ))}
      </div>
    </Shell>
  );
}
