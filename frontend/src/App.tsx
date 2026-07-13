import { Suspense, lazy } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { NavBar } from "./components/NavBar";
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
        <NavBar />

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
