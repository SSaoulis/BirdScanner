import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { Dashboard } from "./pages/Dashboard";
import { History } from "./pages/History";

/**
 * Root application component.
 *
 * Sets up client-side routing with react-router-dom:
 *   /          → Dashboard (live view + system monitor)
 *   /history   → History (filter bar + Timeline / Gallery + bulk download)
 *
 * A top-level navigation bar is rendered on every page.
 */
export function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-900 text-white flex flex-col">
        {/* ── Navigation bar ──────────────────────────────────────── */}
        <nav className="bg-slate-800 border-b border-slate-700 px-6 py-3 flex items-center gap-6">
          <div className="flex items-center gap-2 mr-4">
            <span className="text-xl" role="img" aria-label="bird">🐦</span>
            <span className="font-bold text-lg tracking-tight">BirdScanner</span>
          </div>

          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${
                isActive ? "text-emerald-400" : "text-slate-400 hover:text-white"
              }`
            }
          >
            Dashboard
          </NavLink>

          <NavLink
            to="/history"
            className={({ isActive }) =>
              `text-sm font-medium transition-colors ${
                isActive ? "text-emerald-400" : "text-slate-400 hover:text-white"
              }`
            }
          >
            History
          </NavLink>
        </nav>

        {/* ── Page content ────────────────────────────────────────── */}
        <div className="flex-1">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}
