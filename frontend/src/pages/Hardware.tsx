import { SystemMonitor } from "../components/SystemMonitor";
import { NetworkMonitor } from "../components/NetworkMonitor";

/**
 * Hardware page.
 *
 * Gathers everything about the machine behind the feeder:
 *  - "The Station" resource gauges (CPU / memory / disk / temperature / uptime),
 *    moved here off the Dashboard.
 *  - The network panel: a passive download/upload usage graph with a time-range
 *    toggle, plus an on-demand internet speed test.
 */
export function Hardware() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-8 space-y-8">
      <header>
        <p className="eyebrow mb-2">Behind the feeder</p>
        <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">
          Hardware
        </h1>
        <p className="mt-1 text-sm text-bark">
          How the Pi and its connection are holding up
        </p>
      </header>

      <SystemMonitor />
      <NetworkMonitor />
    </div>
  );
}
