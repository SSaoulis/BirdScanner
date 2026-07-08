import { ResponsiveLine } from "@nivo/line";

import type { TimelineResponse } from "../../api";
import { OTHER_KEY, nivoTheme, seriesColor } from "../../charts/theme";
import { evenlySpaced, shortDate } from "../../charts/format";
import { ChartFrame } from "./ChartFrame";

/** Stacked-area timeline of detections per interval, split by top-N species. */
export function SightingsOverTime({ timeline }: { timeline: TimelineResponse }) {
  const { species, points, interval } = timeline;
  const hasOther = points.some((p) => (p.counts[OTHER_KEY] ?? 0) > 0);
  const keys = hasOther ? [...species, OTHER_KEY] : species;
  const series = keys.map((key) => ({
    id: key,
    data: points.map((p) => ({ x: p.date, y: p.counts[key] ?? 0 })),
  }));
  const tickValues = evenlySpaced(
    points.map((p) => p.date),
    8
  );

  return (
    <ChartFrame
      title="Sightings over time"
      hint={`Detections per ${interval}, stacked by species`}
      height={340}
      isEmpty={points.length === 0}
    >
      <ResponsiveLine
        data={series}
        margin={{ top: 12, right: 20, bottom: 68, left: 44 }}
        xScale={{ type: "point" }}
        yScale={{ type: "linear", min: 0, max: "auto", stacked: true }}
        curve="monotoneX"
        colors={(serie) => seriesColor(String(serie.id), species)}
        enableArea
        areaOpacity={0.85}
        lineWidth={1.5}
        enablePoints={false}
        enableGridX={false}
        enableSlices="x"
        axisLeft={{ tickSize: 0, tickPadding: 8 }}
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
            translateY: 60,
            itemWidth: 96,
            itemHeight: 18,
            symbolSize: 10,
            symbolShape: "circle",
            itemsSpacing: 4,
          },
        ]}
        theme={nivoTheme}
      />
    </ChartFrame>
  );
}
