import { api, timeAgo, type Detection } from "../api";

/**
 * The featured bird's basic stats — the same fields the detection preview card
 * shows in its caption: species, how sure the match is (or a "corrected" mark),
 * the object-detection (YOLO) confidence, and how long ago it was spotted.
 */
function FeaturedStats({ detection }: { detection: Detection }) {
  const { species, confidence, detection_confidence, timestamp } = detection;
  const corrected = detection.corrected === true;
  const confidencePct = (confidence * 100).toFixed(0);
  const detectionPct =
    detection_confidence != null ? (detection_confidence * 100).toFixed(0) : null;

  return (
    <div className="space-y-1">
      <p className="eyebrow">This sighting</p>
      <p
        className="max-w-full truncate font-display text-xl font-medium leading-tight text-ink"
        title={species}
      >
        {species}
      </p>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
        {corrected ? (
          <span className="font-display text-[0.85rem] italic text-gold-deep" title="Species set by you">
            ✎ Corrected
          </span>
        ) : (
          <span className="tnum font-medium text-gold-deep" title="Species-classification confidence">
            {confidencePct}% match
          </span>
        )}
        {!corrected && detectionPct !== null && (
          <span className="tnum text-bark" title="Object-detection confidence (YOLO)">
            · {detectionPct}% spotted
          </span>
        )}
        <span className="text-bark">· {timeAgo(timestamp)}</span>
      </div>
    </div>
  );
}

interface FeaturedDetectionProps {
  /** The featured (most recent) sighting to headline. */
  detection: Detection;
  /** Open the lightbox on the featured detection. */
  onOpenLightbox: () => void;
}

/**
 * The Dashboard hero: the most recent sighting shown at full size on the left,
 * with the featured bird's basic stats on the right — the same species /
 * confidence / time-ago fields the detection preview card shows (via
 * `FeaturedStats`), not the full Lightbox telemetry. Purely presentational —
 * the Dashboard owns the data and lightbox wiring. Rendered only when there is
 * at least one sighting today.
 */
export function FeaturedDetection({ detection, onOpenLightbox }: FeaturedDetectionProps) {
  return (
    <section className="overflow-hidden rounded-xl border border-line bg-card shadow-plate">
      <p className="eyebrow border-b border-line px-5 py-3">Latest visitor</p>
      <div className="grid lg:grid-cols-[minmax(0,1fr)_20rem]">
        {/* Full-size image, centred, opening the lightbox on click. */}
        <button
          type="button"
          onClick={onOpenLightbox}
          aria-label={`Take a closer look at ${detection.species}`}
          className="flex cursor-pointer items-center justify-center bg-paper p-4"
        >
          <img
            src={api.images.fullUrl(detection.id)}
            alt={detection.species}
            loading="eager"
            className="max-h-[60vh] w-auto rounded-lg object-contain shadow-plate"
          />
        </button>

        {/* This bird's basic stats. */}
        <div className="border-t border-line p-5 lg:border-l lg:border-t-0">
          <FeaturedStats detection={detection} />
        </div>
      </div>
    </section>
  );
}
