import { ResponsiveHeatMap } from "@nivo/heatmap";

import type { ActivityCell } from "../../api";
import { HEATMAP_EMPTY, nivoTheme, sequentialColor } from "../../charts/theme";
import { WEEKDAYS, hourColumn } from "../../charts/format";
import { ChartFrame } from "./ChartFrame";

const HOUR_TICKS = [0, 3, 6, 9, 12, 15, 18, 21].map(hourColumn);

/** Hour × day-of-week heatmap (sequential sage ramp) of detection activity. */
export function ActivityHeatmap({ cells }: { cells: ActivityCell[] }) {
  const max = cells.reduce((m, c) => Math.max(m, c.count), 0);
  const grid = WEEKDAYS.map(() => new Array<number>(24).fill(0));
  for (const c of cells) {
    if (c.dow >= 0 && c.dow < 7 && c.hour >= 0 && c.hour < 24) {
      grid[c.dow][c.hour] = c.count;
    }
  }
  const data = WEEKDAYS.map((day, d) => ({
    id: day,
    data: Array.from({ length: 24 }, (_, h) => ({ x: hourColumn(h), y: grid[d][h] })),
  }));

  return (
    <ChartFrame
      title="Weekly rhythm"
      hint="Detections by hour of day and day of week"
      isEmpty={cells.length === 0}
    >
      <ResponsiveHeatMap
        data={data}
        margin={{ top: 28, right: 16, bottom: 12, left: 44 }}
        colors={(cell) =>
          cell.value ? sequentialColor(cell.value / (max || 1)) : HEATMAP_EMPTY
        }
        emptyColor={HEATMAP_EMPTY}
        enableLabels={false}
        xInnerPadding={0.12}
        yInnerPadding={0.12}
        borderRadius={2}
        borderWidth={0}
        axisTop={{ tickSize: 0, tickPadding: 6, tickValues: HOUR_TICKS }}
        axisLeft={{ tickSize: 0, tickPadding: 6 }}
        axisRight={null}
        axisBottom={null}
        hoverTarget="cell"
        theme={nivoTheme}
        tooltip={({ cell }) => (
          <div className="rounded-lg border border-line bg-card px-2 py-1 text-xs text-ink shadow-plate">
            <span className="font-semibold">
              {cell.serieId} {cell.data.x}:00
            </span>
            <span className="text-bark"> · {cell.value ?? 0} sightings</span>
          </div>
        )}
      />
    </ChartFrame>
  );
}
