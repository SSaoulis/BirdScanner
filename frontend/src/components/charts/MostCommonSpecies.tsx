import { ResponsiveBar } from "@nivo/bar";

import type { SpeciesSummary } from "../../api";
import { ACCENT, nivoTheme } from "../../charts/theme";
import { ChartFrame } from "./ChartFrame";

const LIMIT = 12;

/** Sorted horizontal bar chart of the most-frequently-seen species. */
export function MostCommonSpecies({ species }: { species: SpeciesSummary[] }) {
  const top = species.slice(0, LIMIT);
  // API returns count-descending; reverse so the largest bar sits at the top
  // (Nivo draws the first horizontal-bar index at the bottom).
  const data = [...top]
    .reverse()
    .map((s) => ({ species: s.species, count: s.count }));

  return (
    <ChartFrame
      title="Most common species"
      hint={`Top ${Math.min(LIMIT, species.length)} by total sightings`}
      height={Math.max(200, top.length * 30 + 48)}
      isEmpty={species.length === 0}
    >
      <ResponsiveBar
        data={data}
        keys={["count"]}
        indexBy="species"
        layout="horizontal"
        margin={{ top: 8, right: 28, bottom: 36, left: 130 }}
        padding={0.28}
        colors={() => ACCENT}
        borderRadius={4}
        enableGridX
        enableGridY={false}
        axisBottom={{ tickSize: 0, tickPadding: 8 }}
        axisLeft={{ tickSize: 0, tickPadding: 8 }}
        labelSkipWidth={20}
        labelTextColor="#F6F1E2"
        theme={nivoTheme}
        role="img"
        ariaLabel="Most common species by total sightings"
      />
    </ChartFrame>
  );
}
