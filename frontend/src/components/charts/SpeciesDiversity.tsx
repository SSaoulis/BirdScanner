import { ResponsiveLine } from "@nivo/line";

import type { TimelineResponse } from "../../api";
import { ACCENT, nivoTheme } from "../../charts/theme";
import { evenlySpaced, shortDate } from "../../charts/format";
import { ChartFrame } from "./ChartFrame";

/** Line of the number of distinct species seen per interval. */
export function SpeciesDiversity({ timeline }: { timeline: TimelineResponse }) {
  const { points, interval } = timeline;
  const data = [
    {
      id: "distinct species",
      data: points.map((p) => ({ x: p.date, y: p.distinct_species })),
    },
  ];
  const tickValues = evenlySpaced(
    points.map((p) => p.date),
    8
  );

  return (
    <ChartFrame
      title="Species diversity"
      hint={`Distinct species per ${interval}`}
      isEmpty={points.length === 0}
    >
      <ResponsiveLine
        data={data}
        margin={{ top: 12, right: 20, bottom: 40, left: 44 }}
        xScale={{ type: "point" }}
        yScale={{ type: "linear", min: 0, max: "auto" }}
        curve="monotoneX"
        colors={[ACCENT]}
        enableArea
        areaOpacity={0.12}
        pointSize={6}
        pointColor={ACCENT}
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
        theme={nivoTheme}
      />
    </ChartFrame>
  );
}
