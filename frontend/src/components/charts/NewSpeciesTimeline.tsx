import { ResponsiveLine } from "@nivo/line";

import type { FirstSighting } from "../../api";
import { nivoTheme } from "../../charts/theme";
import { evenlySpaced, shortDate } from "../../charts/format";
import { ChartFrame } from "./ChartFrame";

/** Gold accent for the "new species" milestone curve (distinct from the greens). */
const GOLD_DEEP = "#8A6113";

/** Cumulative step curve of distinct species first seen over time (all-time). */
export function NewSpeciesTimeline({ sightings }: { sightings: FirstSighting[] }) {
  // The API returns first-sightings oldest-first, so the running index is the
  // cumulative distinct-species count. Keep a y → species lookup for tooltips.
  const speciesAt = new Map<number, string>();
  const data = [
    {
      id: "species",
      data: sightings.map((s, i) => {
        speciesAt.set(i + 1, s.species);
        return { x: s.first_seen.slice(0, 10), y: i + 1 };
      }),
    },
  ];
  const tickValues = evenlySpaced(
    sightings.map((s) => s.first_seen.slice(0, 10)),
    8
  );

  return (
    <ChartFrame
      title="New species over time"
      hint="Cumulative count of distinct species first seen"
      isEmpty={sightings.length === 0}
    >
      <ResponsiveLine
        data={data}
        margin={{ top: 12, right: 24, bottom: 40, left: 44 }}
        xScale={{ type: "point" }}
        yScale={{ type: "linear", min: 0, max: "auto" }}
        curve="stepAfter"
        colors={[GOLD_DEEP]}
        enableArea
        areaOpacity={0.1}
        pointSize={7}
        pointColor={GOLD_DEEP}
        pointBorderWidth={0}
        enableGridX={false}
        axisLeft={{
          tickSize: 0,
          tickPadding: 8,
          format: (v) => (Number.isInteger(Number(v)) ? String(v) : ""),
        }}
        axisBottom={{
          tickSize: 0,
          tickPadding: 8,
          tickValues,
          format: shortDate,
        }}
        useMesh
        tooltip={({ point }) => {
          const n = Number(point.data.y);
          return (
            <div className="rounded-lg border border-line bg-card px-2 py-1 text-xs text-ink shadow-plate">
              <span className="font-semibold">#{n}</span>
              <span className="text-bark"> · {speciesAt.get(n) ?? ""}</span>
            </div>
          );
        }}
        theme={nivoTheme}
      />
    </ChartFrame>
  );
}
