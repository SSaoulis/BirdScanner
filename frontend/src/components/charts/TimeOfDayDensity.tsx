import type { ReactNode } from "react";
import { ResponsiveLine } from "@nivo/line";

import type { TimeBin } from "../../api";
import { ACCENT, circularGaussianSmooth, nivoTheme } from "../../charts/theme";
import { hourLabel } from "../../charts/format";
import { ChartFrame } from "./ChartFrame";

interface TimeOfDayDensityProps {
  bins: TimeBin[];
  binMinutes: number;
  /** Optional species picker rendered in the card header. */
  speciesControl?: ReactNode;
  isEmpty?: boolean;
}

const X_TICKS = [0, 180, 360, 540, 720, 900, 1080, 1260, 1440];

/**
 * Time-of-day density curve: a fine histogram from the API, densified and
 * circularly Gaussian-smoothed on the client so the 24-hour clock reads as a
 * continuous curve (with the midnight wrap handled).
 */
export function TimeOfDayDensity({
  bins,
  binMinutes,
  speciesControl,
  isEmpty,
}: TimeOfDayDensityProps) {
  const count = Math.max(1, Math.round(1440 / binMinutes));
  const raw = new Array<number>(count).fill(0);
  for (const b of bins) {
    const idx = Math.min(count - 1, Math.max(0, Math.floor(b.minute / binMinutes)));
    raw[idx] += b.count;
  }
  const smooth = circularGaussianSmooth(raw);
  const data = [
    { id: "detections", data: smooth.map((y, i) => ({ x: i * binMinutes, y })) },
  ];

  return (
    <ChartFrame
      title="Time of day"
      hint="When birds visit across the 24-hour clock"
      action={speciesControl}
      isEmpty={isEmpty ?? bins.length === 0}
    >
      <ResponsiveLine
        data={data}
        margin={{ top: 12, right: 20, bottom: 40, left: 16 }}
        xScale={{ type: "linear", min: 0, max: 1440 }}
        yScale={{ type: "linear", min: 0, max: "auto" }}
        curve="basis"
        colors={[ACCENT]}
        enableArea
        areaOpacity={0.16}
        enablePoints={false}
        enableGridX={false}
        enableGridY={false}
        axisLeft={null}
        axisBottom={{
          tickSize: 0,
          tickPadding: 8,
          tickValues: X_TICKS,
          format: (v) => hourLabel(Number(v)),
        }}
        enableSlices="x"
        sliceTooltip={({ slice }) => {
          const minute = Number(slice.points[0].data.x);
          const idx = Math.min(count - 1, Math.floor(minute / binMinutes));
          return (
            <div className="rounded-lg border border-line bg-card px-2 py-1 text-xs text-ink shadow-plate">
              <span className="font-semibold">{hourLabel(minute)}</span>
              <span className="text-bark"> · {raw[idx]} sightings</span>
            </div>
          );
        }}
        theme={nivoTheme}
      />
    </ChartFrame>
  );
}
