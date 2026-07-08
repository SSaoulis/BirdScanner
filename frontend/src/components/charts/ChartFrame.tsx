import type { ReactNode } from "react";

interface ChartFrameProps {
  /** Chart heading (Fraunces display). */
  title: string;
  /** Optional one-line description under the title. */
  hint?: string;
  /** Fixed height of the plotting area in px (charts need explicit height). */
  height?: number;
  /** When true, render the empty state instead of the chart. */
  isEmpty?: boolean;
  /** Message shown in the empty state. */
  emptyLabel?: string;
  /** Optional controls rendered in the header (e.g. a species picker). */
  action?: ReactNode;
  children: ReactNode;
}

/**
 * Card shell shared by every Statistics chart: a titled `bg-card` plate with a
 * fixed-height plotting area and a graceful empty state, matching the History /
 * Hardware card styling.
 */
export function ChartFrame({
  title,
  hint,
  height = 280,
  isEmpty = false,
  emptyLabel = "No sightings in this range yet.",
  action,
  children,
}: ChartFrameProps) {
  return (
    <section className="rounded-2xl border border-line bg-card p-4 shadow-plate">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="font-display text-lg font-semibold tracking-tight text-ink">
            {title}
          </h2>
          {hint && <p className="mt-0.5 text-xs text-bark">{hint}</p>}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
      <div style={{ height }}>
        {isEmpty ? (
          <div className="flex h-full items-center justify-center text-sm text-bark">
            {emptyLabel}
          </div>
        ) : (
          children
        )}
      </div>
    </section>
  );
}
