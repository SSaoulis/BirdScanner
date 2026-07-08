import { ResponsiveLine } from "@nivo/line";

import type { DayWindow } from "../../api";
import { DUSK, SUNRISE, nivoTheme } from "../../charts/theme";
import { evenlySpaced, hourLabel, minuteToClock, shortDate } from "../../charts/format";
import { ChartFrame } from "./ChartFrame";

const Y_TICKS = [0, 360, 720, 1080, 1440];

/** Earliest ("first seen") and latest ("last seen") detection time per day. */
export function DailyActivityWindow({ days }: { days: DayWindow[] }) {
  const data = [
    { id: "First seen", data: days.map((d) => ({ x: d.date, y: d.first_minute })) },
    { id: "Last seen", data: days.map((d) => ({ x: d.date, y: d.last_minute })) },
  ];
  const tickValues = evenlySpaced(
    days.map((d) => d.date),
    8
  );

  return (
    <ChartFrame
      title="Daily activity window"
      hint="Earliest and latest sighting each day"
      isEmpty={days.length === 0}
    >
      <ResponsiveLine
        data={data}
        margin={{ top: 12, right: 20, bottom: 60, left: 52 }}
        xScale={{ type: "point" }}
        yScale={{ type: "linear", min: 0, max: 1440 }}
        curve="monotoneX"
        colors={[SUNRISE, DUSK]}
        pointSize={5}
        pointBorderWidth={0}
        enableGridX={false}
        yFormat={(v) => minuteToClock(Number(v))}
        axisLeft={{
          tickSize: 0,
          tickPadding: 8,
          tickValues: Y_TICKS,
          format: (v) => hourLabel(Number(v)),
        }}
        axisBottom={{
          tickSize: 0,
          tickPadding: 8,
          tickValues,
          format: shortDate,
        }}
        legends={[
          {
            anchor: "bottom",
            direction: "row",
            translateY: 52,
            itemWidth: 96,
            itemHeight: 18,
            symbolSize: 10,
            symbolShape: "circle",
            itemsSpacing: 4,
          },
        ]}
        useMesh
        theme={nivoTheme}
      />
    </ChartFrame>
  );
}
