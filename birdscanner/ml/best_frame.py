"""Per-track best-frame selection for detection stills.

While a track is alive the camera callback observes it once per frame; this
helper keeps only the single highest-scoring frame seen for each track so that,
when the track becomes stable and classification fires, the saved still and
thumbnail come from the clearest frame rather than the (arbitrary) frame that
happened to trigger classification.

The selector is deliberately platform-independent (pure numpy) so it lives in
``ml/`` and never pulls in picamera2. It is written on the camera thread and
read on the classifier thread, so all access is guarded by a lock.
"""

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BestCandidate:
    """The best frame retained for a single track.

    Attributes:
        frame: The full RGB frame the candidate was observed in.
        box: The detection box ``(x, y, w, h)`` in that frame's pixel coordinates.
        score: The score that won this frame the "best" slot (higher is better).
    """

    frame: np.ndarray
    box: tuple
    score: float


class BestFrameSelector:
    """Keeps the single highest-scoring frame observed per track.

    Thread-safe: ``observe`` runs on the camera callback thread while ``take``
    runs on the classification worker thread, so every access holds ``_lock``.
    At most one frame is retained per active track, bounding memory use.
    """

    def __init__(self) -> None:
        """Initialise an empty selector."""
        self._lock = threading.Lock()
        self._best: dict[int, BestCandidate] = {}

    def observe(
        self, track_id: int, frame: np.ndarray, box: tuple, score: float
    ) -> None:
        """Offer a frame as the possible new best for a track.

        The frame reference is stored as-is (not copied); callers must pass a
        frame that will not be mutated afterwards — the pipeline passes the
        per-frame ``full_img`` copy, which nothing mutates in place.

        Args:
            track_id: Identifier of the track the detection belongs to.
            frame: The full RGB frame the detection was seen in.
            box: The detection box ``(x, y, w, h)`` in ``frame`` pixel coordinates.
            score: Quality score for this frame; the highest-scoring frame wins.
        """
        with self._lock:
            current = self._best.get(track_id)
            if current is None or score > current.score:
                self._best[track_id] = BestCandidate(frame=frame, box=box, score=score)

    def take(self, track_id: int) -> Optional[BestCandidate]:
        """Remove and return the best candidate for a track, if any.

        Args:
            track_id: Identifier of the track to take the best frame for.

        Returns:
            The retained :class:`BestCandidate`, or ``None`` if the track was
            never observed.
        """
        with self._lock:
            return self._best.pop(track_id, None)

    def discard(self, track_id: int) -> None:
        """Drop any retained frame for a track, freeing its memory.

        Safe to call for an unknown ``track_id``. Wire this into the tracker's
        ``on_track_deleted`` callback so ended tracks do not leak frames.

        Args:
            track_id: Identifier of the track to discard.
        """
        with self._lock:
            self._best.pop(track_id, None)
