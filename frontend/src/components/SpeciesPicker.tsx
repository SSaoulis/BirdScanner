import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../api";

interface SpeciesPickerProps {
  /** The species currently on the record (marked "on record", not selectable). */
  current: string;
  /** Called with the chosen species when the user commits a correction. */
  onConfirm: (species: string) => void;
  /** Called when the user backs out without correcting. */
  onCancel: () => void;
  /** True while a correction request is in flight (locks the list). */
  busy?: boolean;
  /** An error to show inline (e.g. a rejected correction). */
  errorMessage?: string | null;
}

/** Load state for the classifier vocabulary. */
type VocabState =
  | { kind: "loading" }
  | { kind: "ready"; species: string[] }
  | { kind: "offline" }
  | { kind: "error"; message: string };

// The vocabulary is a fixed ~700-entry list, so fetch it once and reuse it for
// every picker opened this session.
let vocabCache: string[] | null = null;

/**
 * Inline type-ahead for re-identifying a detection's species.
 *
 * Reads like writing a correction in a field guide's margin: search the guide's
 * species list, pick the right one, and it replaces the printed ID. The list is
 * the classifier's own vocabulary (`api.species.vocabulary`), so a correction can
 * only ever be a species the model knows — which keeps the saved folders and
 * retraining buckets aligned. Fully keyboard-driven (type to filter, ↑/↓ to move,
 * Enter to choose, Esc to cancel).
 */
export function SpeciesPicker({
  current,
  onConfirm,
  onCancel,
  busy = false,
  errorMessage = null,
}: SpeciesPickerProps) {
  const [vocab, setVocab] = useState<VocabState>(
    vocabCache ? { kind: "ready", species: vocabCache } : { kind: "loading" }
  );
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);

  // Load the vocabulary once (from cache when already fetched this session).
  useEffect(() => {
    if (vocabCache) return;
    let cancelled = false;
    api.species
      .vocabulary()
      .then((species) => {
        vocabCache = species;
        if (!cancelled) setVocab({ kind: "ready", species });
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        // 503 = the detector (which owns the class map) is offline. Corrections
        // need it too, so treat it as a distinct "come back when it's up" state.
        if (e instanceof ApiError && e.status === 503) {
          setVocab({ kind: "offline" });
        } else {
          setVocab({
            kind: "error",
            message: e instanceof Error ? e.message : "Couldn’t load the species list",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Focus the search field as soon as the picker opens.
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const allSpecies = vocab.kind === "ready" ? vocab.species : [];
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    const pool = q
      ? allSpecies.filter((s) => s.toLowerCase().includes(q))
      : allSpecies;
    // Cap the rendered rows so a blank query doesn't paint all ~700 at once.
    return pool.slice(0, 60);
  }, [query, allSpecies]);

  // Keep the active row in range and scrolled into view as the list changes.
  useEffect(() => {
    setActive(0);
  }, [query]);
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const el = list.children[active] as HTMLElement | undefined;
    el?.scrollIntoView({ block: "nearest" });
  }, [active, matches]);

  function commit(species: string) {
    if (busy) return;
    if (species === current) {
      onCancel();
      return;
    }
    onConfirm(species);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => Math.min(i + 1, matches.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const chosen = matches[active];
      if (chosen) commit(chosen);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    }
  }

  return (
    <div
      className="w-72 rounded-xl border border-line bg-card p-3 shadow-plate-lift"
      onClick={(e) => e.stopPropagation()}
      role="dialog"
      aria-label="Correct species"
    >
      <div className="mb-2 flex items-center justify-between">
        <span className="eyebrow after:hidden">Re-identify</span>
        <button
          className="text-xs font-medium text-bark transition-colors hover:text-ink"
          onClick={onCancel}
          aria-label="Cancel correction"
        >
          Cancel
        </button>
      </div>

      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Search the guide…"
        aria-label="Search species"
        disabled={busy || vocab.kind !== "ready"}
        className="w-full rounded-lg border border-line bg-paper px-3 py-2 text-sm text-ink placeholder:text-bark/70 focus:border-gold focus:outline-none focus:ring-2 focus:ring-gold disabled:opacity-60"
      />

      <div className="mt-2">
        {vocab.kind === "loading" && (
          <p className="px-1 py-6 text-center text-sm text-bark">Opening the guide…</p>
        )}
        {vocab.kind === "offline" && (
          <p className="px-1 py-6 text-center text-sm text-bark">
            The species list is offline right now — the detector isn’t reachable.
          </p>
        )}
        {vocab.kind === "error" && (
          <p className="px-1 py-6 text-center text-sm text-rust">{vocab.message}</p>
        )}

        {vocab.kind === "ready" && matches.length === 0 && (
          <p className="px-1 py-6 text-center text-sm text-bark">
            No species matches “{query.trim()}”.
          </p>
        )}

        {vocab.kind === "ready" && matches.length > 0 && (
          <ul
            ref={listRef}
            className="max-h-56 overflow-y-auto"
            role="listbox"
            aria-label="Species"
          >
            {matches.map((species, i) => {
              const isCurrent = species === current;
              return (
                <li key={species} role="option" aria-selected={i === active}>
                  <button
                    className={`flex w-full items-center justify-between gap-2 rounded-md px-2.5 py-1.5 text-left font-display text-sm transition-colors ${
                      i === active ? "bg-gold/15 text-ink" : "text-ink hover:bg-paper"
                    } ${isCurrent ? "cursor-default text-bark" : ""}`}
                    onMouseEnter={() => setActive(i)}
                    onClick={() => commit(species)}
                    disabled={busy || isCurrent}
                  >
                    <span>{species}</span>
                    {isCurrent && (
                      <span className="shrink-0 font-sans text-[10px] uppercase tracking-wide text-bark">
                        on record
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {busy && (
        <p className="mt-2 px-1 text-xs text-bark" role="status">
          Saving correction…
        </p>
      )}
      {errorMessage && !busy && (
        <p className="mt-2 px-1 text-xs text-rust" role="alert">
          {errorMessage}
        </p>
      )}
    </div>
  );
}
