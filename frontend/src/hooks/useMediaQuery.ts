import { useEffect, useState } from "react";

/**
 * Subscribe to a CSS media query and return whether it currently matches.
 *
 * Re-renders the caller whenever the match state changes (e.g. the viewport
 * crosses a breakpoint or the device rotates). SSR-safe: defaults to `false`
 * when `window` is unavailable, resolving to the real value after mount.
 *
 * @param query A media-query string, e.g. `"(min-width: 1024px)"`.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.matchMedia(query).matches : false
  );

  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches);
    // Sync immediately in case the query changed between renders.
    setMatches(mql.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

/**
 * Whether the viewport is at least Tailwind's `lg` breakpoint (1024px) — the
 * width above which the app uses its wide, side-by-side layouts. Defaults to
 * `true` on the server so the desktop markup renders first, then corrects on
 * mount for narrow clients.
 */
export function useIsDesktop(): boolean {
  const [isDesktop, setIsDesktop] = useState<boolean>(() =>
    typeof window !== "undefined"
      ? window.matchMedia("(min-width: 1024px)").matches
      : true
  );

  useEffect(() => {
    const mql = window.matchMedia("(min-width: 1024px)");
    const onChange = (e: MediaQueryListEvent) => setIsDesktop(e.matches);
    setIsDesktop(mql.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  return isDesktop;
}
