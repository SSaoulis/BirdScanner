import type { ReactNode } from "react";
import { useLayoutEffect, useState } from "react";
import {
  LazyMotion,
  domMax,
  m,
  animate,
  useMotionValue,
  useReducedMotion,
  type PanInfo,
} from "motion/react";
import { api, type Detection } from "../api";
import { PlateControlBar, SpecimenLabel, MobilePanelTabs } from "./LightboxChrome";

interface MobileSwipeStripProps {
  /** The live, interactive centre card (the current sighting). */
  children: ReactNode;
  /** Id of the centred detection; drives the recentre after a committed swipe. */
  currentId: number;
  /** Neighbour sightings shown in the side slots. Null at a list end. */
  prevDetection: Detection | null;
  nextDetection: Detection | null;
  /** Neighbour positions, for the "n / total" chip on each preview. Null ends. */
  prevPosition: { index: number; total: number } | null;
  nextPosition: { index: number; total: number } | null;
  /** Commit funnels — navigate the parent's list. Null at a list end. */
  onPrev: (() => void) | null;
  onNext: (() => void) | null;
  /** False in video mode / while correcting, so the drag never hijacks them. */
  enabled: boolean;
}

/**
 * Mobile-only, finger-tracking swipe navigation for the lightbox. Lazily loaded
 * (it pulls in `motion`), so desktop and the initial bundle never pay for it.
 *
 * Lays the current card as the centre slot of a three-up filmstrip
 * `[prev | current | next]`, each slot one viewport wide, shifted left one slot
 * so the centre rests on screen. A horizontal drag tracks the finger (`x`),
 * sliding the current plate out and the neighbour in; releasing past a distance
 * or velocity threshold commits to that neighbour (via `onPrev`/`onNext`), else
 * it rubber-bands back. After a commit the just-slid-in plate becomes the new
 * centre slot, so `x` is jumped back to 0 (invisible — identical content either
 * side). Reduced-motion callers get an instant snap instead of the spring.
 */
export default function MobileSwipeStrip({
  children,
  currentId,
  prevDetection,
  nextDetection,
  prevPosition,
  nextPosition,
  onPrev,
  onNext,
  enabled,
}: MobileSwipeStripProps) {
  const x = useMotionValue(0);
  const reduceMotion = useReducedMotion();
  // Neighbour plates start as the (already-cached) thumbnail and upgrade to the
  // full-res still the moment a drag begins, so the incoming plate is crisp by
  // the time the finger has moved — without paying 3× full-res on open.
  const [hires, setHires] = useState(false);

  const canPrev = prevDetection !== null && onPrev !== null;
  const canNext = nextDetection !== null && onNext !== null;
  const swipeable = enabled && (canPrev || canNext);

  // Recentre whenever the committed detection changes. useLayoutEffect (before
  // paint) so the jump lands on the frame the new slots render — otherwise a
  // stale `x` would flash the wrong (shifted) neighbour for a frame.
  useLayoutEffect(() => {
    x.jump(0);
  }, [currentId, x]);

  /**
   * Animate `x` to `target`; on arrival, run `commit` (the nav callback). A
   * *committing* settle uses a short, bounded tween so `onComplete` — which
   * swaps the real interactive card in for the inert neighbour preview — fires
   * promptly (a full-width spring's long settle tail left the controls
   * un-tappable for ~1s). The rubber-band-back (no commit) keeps the spring so a
   * cancelled half-swipe still feels elastic.
   */
  function settle(target: number, commit: (() => void) | null) {
    animate(x, target, {
      ...(reduceMotion
        ? { duration: 0 }
        : commit
          ? { type: "tween", duration: 0.26, ease: "easeOut" }
          : { type: "spring", stiffness: 550, damping: 45 }),
      onComplete: commit ?? undefined,
    });
  }

  /**
   * Decide commit-vs-rubber-band from the release distance and velocity. Both
   * the distance and the flick tests also require the plate to actually be on
   * that side (`offset.x` sign), so a drag-partway-then-flick-back gesture
   * cancels instead of committing to the opposite neighbour.
   */
  function handleDragEnd(_: PointerEvent, info: PanInfo) {
    const width = window.innerWidth;
    const distance = width * 0.35;
    const speed = 450;
    const { offset, velocity } = info;
    const wantsNext = offset.x < 0 && (offset.x <= -distance || velocity.x <= -speed);
    const wantsPrev = offset.x > 0 && (offset.x >= distance || velocity.x >= speed);
    if (wantsNext && canNext) {
      settle(-width, onNext);
    } else if (wantsPrev && canPrev) {
      settle(width, onPrev);
    } else {
      settle(0, null);
    }
  }

  // Constraints in px so an over-drag toward an existing neighbour tracks the
  // finger, while a missing neighbour (bound 0) resists with only slight
  // elastic. Read at render; navigation re-renders the strip anyway.
  const width = typeof window !== "undefined" ? window.innerWidth : 0;

  return (
    <LazyMotion features={domMax} strict>
      <m.div
        className="flex w-[300vw] -ml-[100vw] items-start"
        style={{ x }}
        drag={swipeable ? "x" : false}
        dragConstraints={{ left: canNext ? -width : 0, right: canPrev ? width : 0 }}
        dragElastic={0.16}
        dragMomentum={false}
        onDragStart={() => setHires(true)}
        onDragEnd={handleDragEnd}
      >
        <div
          className="flex w-screen shrink-0 items-start justify-center px-4 py-4"
          aria-hidden="true"
        >
          {prevDetection && (
            <NeighborPreview
              detection={prevDetection}
              position={prevPosition}
              hires={hires}
            />
          )}
        </div>
        <div className="flex w-screen shrink-0 items-start justify-center px-4 py-4">
          {/* Stop propagation so a tap on the card never closes; taps in the
              empty side area still bubble to the backdrop to close. */}
          <div onClick={(e) => e.stopPropagation()}>{children}</div>
        </div>
        <div
          className="flex w-screen shrink-0 items-start justify-center px-4 py-4"
          aria-hidden="true"
        >
          {nextDetection && (
            <NeighborPreview
              detection={nextDetection}
              position={nextPosition}
              hires={hires}
            />
          )}
        </div>
      </m.div>
    </LazyMotion>
  );
}

interface NeighborPreviewProps {
  /** The neighbouring sighting to preview. */
  detection: Detection;
  /** The neighbour's position, for the "n / total" chip. Null at a list end. */
  position: { index: number; total: number } | null;
  /** When true, load the full-res still instead of the cached thumbnail. */
  hires: boolean;
}

/**
 * A non-interactive preview of a neighbouring sighting for the swipe filmstrip.
 * It mirrors the *live mobile card's* full layout — the image plate with its
 * control bar (Photo/Video, box, Close), the specimen-label caption (Correct-ID,
 * Download, Delete) and the Field-guide / Advanced-stats switcher — rendered
 * inert (`interactive={false}`, whole subtree `pointer-events-none`) in each
 * control's default state. Because a freshly-committed card resets to those same
 * defaults, the overlay chrome is already on screen as the plate slides in and
 * the inert→live swap on commit shows no pop-in. It still carries no panels or
 * network fetches; the neighbour only "wakes up" into the full interactive card
 * once a swipe commits it to the centre.
 */
function NeighborPreview({ detection, position, hires }: NeighborPreviewProps) {
  const src = hires
    ? api.images.fullUrl(detection.id)
    : api.images.thumbnailUrl(detection.id);
  const hasBox =
    detection.box_x !== null &&
    detection.box_y !== null &&
    detection.box_w !== null &&
    detection.box_h !== null;
  const detectionPct =
    detection.detection_confidence != null
      ? (detection.detection_confidence * 100).toFixed(1)
      : null;

  return (
    <div className="pointer-events-none flex flex-col items-start gap-3">
      <div className="relative w-fit">
        <img
          src={src}
          alt=""
          decoding="async"
          className="block max-h-[60vh] max-w-full rounded-lg bg-ink shadow-plate-lift"
        />
        {/* Box coords are full-frame; the resting thumbnail is a crop of the
            bird ROI, so the box only lines up once the full still is showing. */}
        {hasBox && hires && (
          <div
            className="pointer-events-none absolute rounded-sm border-2 border-gold shadow-[0_0_0_1px_rgba(0,0,0,0.45)]"
            style={{
              left: `${detection.box_x! * 100}%`,
              top: `${detection.box_y! * 100}%`,
              width: `${detection.box_w! * 100}%`,
              height: `${detection.box_h! * 100}%`,
            }}
          />
        )}
        <PlateControlBar
          position={position}
          mode="photo"
          hasVideo={detection.video_path !== null}
          hasBox={hasBox}
          showBox
          noVideoReason={detection.no_video_reason}
          interactive={false}
        />
      </div>
      <SpecimenLabel
        species={detection.species}
        confidencePct={(detection.confidence * 100).toFixed(1)}
        detectionPct={detectionPct}
        corrected={detection.corrected === true}
        originalSpecies={detection.original_species}
        timestamp={detection.timestamp}
        downloadUrl={api.images.fullUrl(detection.id)}
        interactive={false}
      />
      <div className="w-full">
        <MobilePanelTabs showReference={false} showStats={false} interactive={false} />
      </div>
    </div>
  );
}
