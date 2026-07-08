import { Suspense, lazy } from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { Dashboard } from "./pages/Dashboard";
import { History } from "./pages/History";
import { Camera } from "./pages/Camera";
import { Hardware } from "./pages/Hardware";
import { Settings } from "./pages/Settings";

// Code-split the Statistics page: it pulls in Nivo (~400 kB), which no other
// route needs, so it loads on demand rather than bloating the initial bundle.
const Statistics = lazy(() =>
  import("./pages/Statistics").then((m) => ({ default: m.Statistics }))
);

/** A simple naturalist's feather mark, drawn rather than emoji'd. */
function FeatherMark() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-7 w-7 text-gold"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M20 4C9 6 5 12 5 18l3-1 11-13Z" />
      <path d="M5 18l5-5" />
      <path d="M9 8.5c2.2.4 4.2.4 6-.5" />
      <path d="M8 12.5c2.4.5 4.6.4 6.5-.6" />
    </svg>
  );
}

const navLinkClass = ({ isActive }: { isActive: boolean }): string =>
  [
    "relative pb-1 text-sm font-medium transition-colors",
    "after:absolute after:inset-x-0 after:-bottom-px after:h-0.5 after:rounded-full after:transition-colors",
    isActive
      ? "text-ink after:bg-gold"
      : "text-bark hover:text-ink after:bg-transparent",
  ].join(" ");

/**
 * Root application component.
 *
 * Sets up client-side routing with react-router-dom:
 *   /          → Dashboard (today's sightings + station vitals)
 *   /history   → History (filter bar + Timeline / Gallery + bulk download)
 *   /camera    → Camera (on-demand snapshot + detection-region editor)
 *   /hardware  → Hardware (station vitals + network usage graph + speed test)
 *   /settings  → Settings (runtime detection/saving/video/system parameters)
 *
 * A journal masthead is rendered on every page.
 */
export function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-paper text-ink flex flex-col">
        {/* ── Masthead ─────────────────────────────────────────────── */}
        <nav className="sticky top-0 z-40 border-b border-line bg-paper/80 backdrop-blur supports-[backdrop-filter]:bg-paper/70">
          <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-6 py-4">
            <div className="flex items-center gap-3">
              <FeatherMark />
              <div className="leading-none">
                <span className="block font-display text-xl font-semibold tracking-tight text-ink">
                  BirdScanner
                </span>
                <span className="mt-1 block text-[0.65rem] font-semibold uppercase tracking-[0.22em] text-sage-deep">
                  Backyard field journal
                </span>
              </div>
            </div>

            <div className="flex items-center gap-7">
              <NavLink to="/" end className={navLinkClass}>
                Today
              </NavLink>
              <NavLink to="/history" className={navLinkClass}>
                Sightings
              </NavLink>
              <NavLink to="/stats" className={navLinkClass}>
                Statistics
              </NavLink>
              <NavLink to="/camera" className={navLinkClass}>
                Camera
              </NavLink>
              <NavLink to="/hardware" className={navLinkClass}>
                Hardware
              </NavLink>
              <NavLink to="/settings" className={navLinkClass}>
                Settings
              </NavLink>
            </div>
          </div>
        </nav>

        {/* ── Page content ────────────────────────────────────────── */}
        <div className="flex-1">
          <Suspense
            fallback={
              <div className="flex h-64 items-center justify-center text-sm text-bark">
                Loading…
              </div>
            }
          >
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/history" element={<History />} />
              <Route path="/stats" element={<Statistics />} />
              <Route path="/camera" element={<Camera />} />
              <Route path="/hardware" element={<Hardware />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Suspense>
        </div>
      </div>
    </BrowserRouter>
  );
}
