import { useEffect, useRef, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";

/** A single navigation destination. */
interface NavItem {
  /** Route path. */
  to: string;
  /** Label shown in the nav and the mobile menu button. */
  label: string;
  /** Whether the route should match exactly (used for the index route). */
  end?: boolean;
}

/**
 * The masthead's navigation destinations, defined once so the inline (desktop)
 * links and the collapsed (mobile) dropdown never drift out of sync.
 */
const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Today", end: true },
  { to: "/history", label: "Sightings" },
  { to: "/stats", label: "Statistics" },
  { to: "/camera", label: "Camera" },
  { to: "/hardware", label: "Hardware" },
  { to: "/settings", label: "Settings" },
];

/** A naturalist's feather mark, drawn rather than emoji'd. */
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
 * Match the current location to a nav item so the collapsed menu button can
 * name the page the reader is on. Falls back to the index route's label.
 */
function currentLabel(pathname: string): string {
  const match = NAV_ITEMS.find((item) =>
    item.end ? pathname === item.to : pathname.startsWith(item.to)
  );
  return match?.label ?? NAV_ITEMS[0].label;
}

/**
 * The journal masthead: brand block plus navigation. On wide screens the
 * destinations sit inline; below `lg` they collapse into a single dropdown
 * (labelled with the current page) so they never overflow a phone's width.
 */
export function NavBar() {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  // Close the mobile menu on outside click or Escape.
  useEffect(() => {
    if (!menuOpen) return;
    function handlePointer(e: PointerEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setMenuOpen(false);
    }
    window.addEventListener("pointerdown", handlePointer);
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("pointerdown", handlePointer);
      window.removeEventListener("keydown", handleKey);
    };
  }, [menuOpen]);

  // Any route change closes the menu (covers link taps and browser navigation).
  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  return (
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

        {/* Inline destinations — desktop only. */}
        <div className="hidden items-center gap-7 lg:flex">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end} className={navLinkClass}>
              {item.label}
            </NavLink>
          ))}
        </div>

        {/* Collapsed dropdown — below lg. Labelled with the current page. */}
        <div className="relative lg:hidden" ref={menuRef}>
          <button
            type="button"
            className="flex items-center gap-2 rounded-lg border border-line bg-card px-3 py-1.5 text-sm font-medium text-ink shadow-plate transition-colors hover:bg-paper"
            onClick={() => setMenuOpen((v) => !v)}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            aria-label="Open navigation menu"
          >
            {currentLabel(location.pathname)}
            <svg
              viewBox="0 0 12 12"
              className={`h-3 w-3 text-bark transition-transform ${menuOpen ? "rotate-180" : ""}`}
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              aria-hidden="true"
            >
              <path d="M2.5 4.5L6 8l3.5-3.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>

          {menuOpen && (
            <div
              className="absolute right-0 top-full z-50 mt-2 min-w-[11rem] overflow-hidden rounded-xl border border-line bg-card py-1 shadow-plate-lift"
              role="menu"
            >
              {NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  role="menuitem"
                  className={({ isActive }) =>
                    [
                      "block px-4 py-2 text-sm transition-colors",
                      isActive
                        ? "bg-paper font-semibold text-ink"
                        : "text-bark hover:bg-paper hover:text-ink",
                    ].join(" ")
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}
