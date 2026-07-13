import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type ExpectedSpecies, type ExpectedThisWeek as ExpectedData } from "../api";

// How many species the panel asks for — small on purpose so the rail stays a
// glanceable field-guide index rather than a long list.
const EXPECTED_COUNT = 6;

/**
 * Small square thumbnail for an expected species, pulled from the offline
 * reference bank's cached thumbnail (a few-KB rendition, not the full-res
 * original). When the bank has no image for the species (or isn't deployed) the
 * request 404s and we fall back to a monogram tile, so the row never shifts.
 */
function SpeciesThumb({ species }: { species: string }) {
  const [failed, setFailed] = useState(false);
  const monogram = species.trim().charAt(0).toUpperCase() || "·";

  if (failed) {
    return (
      <div
        className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-md bg-sage/20 font-display text-sm text-sage-deep"
        aria-hidden="true"
      >
        {monogram}
      </div>
    );
  }
  return (
    <img
      src={api.species.referenceThumbnailUrl(species)}
      alt=""
      width={36}
      height={36}
      decoding="async"
      className="h-9 w-9 flex-shrink-0 rounded-md object-cover shadow-plate"
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

/**
 * One field-guide index row: a small thumbnail and the common name in the
 * journal serif. Rows are rendered in the geomodel's ranked order (most to
 * least expected), so their order alone carries the "how likely" cue — no
 * probability indicator is shown, since the scores cluster high and reading
 * them as a scale is more confusing than helpful.
 */
function ExpectedRow({ item }: { item: ExpectedSpecies }) {
  return (
    <li className="flex items-center gap-3">
      <SpeciesThumb species={item.species} />
      <span
        className="min-w-0 flex-1 truncate font-display text-sm text-ink"
        title={item.species}
      >
        {item.species}
      </span>
    </li>
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
      .expected(EXPECTED_COUNT)
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
    <section className="rounded-xl border border-line bg-card p-4 shadow-plate">
      <p className="eyebrow mb-3">In season · this week</p>
      {children}
    </section>
  );

  if (loading) {
    return (
      <Shell>
        <div className="animate-pulse space-y-2.5">
          {Array.from({ length: EXPECTED_COUNT }).map((_, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="h-9 w-9 flex-shrink-0 rounded-md bg-line/60" />
              <div className="h-3 flex-1 rounded bg-line/60" />
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

  return (
    <Shell>
      <ul className="space-y-2.5">
        {data.species.map((item) => (
          <ExpectedRow key={item.species} item={item} />
        ))}
      </ul>
    </Shell>
  );
}
