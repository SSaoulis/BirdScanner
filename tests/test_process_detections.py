from src.object_detection import (
    StableDetectionTracker,
    StableTrack,
    should_classify_track,
    should_run_bird_classification_for_detection,
    update_tracks_for_frame,
)


class DummyDet:
    def __init__(self, box):
        self.box = box


def _stable_set_after_update(tracker: StableDetectionTracker) -> set[int]:
    """Helper: after `update_frame`, return which detection_ids are allowed to classify."""

    stable = set()
    # Detection IDs are per-frame indices.
    # This helper simply queries them all (0..len-1) based on the last frame mapping.
    for det_id in list(tracker._last_per_det_track.keys()):
        if should_run_bird_classification_for_detection(det_id, tracker=tracker):
            stable.add(det_id)
    return stable


def _boxes_for_frame(*boxes: tuple) -> list[DummyDet]:
    """Helper: build a list of DummyDet from a list of boxes."""

    return [DummyDet(b) for b in boxes]


def test_update_tracks_for_frame_increments_stability():
    tracks = []
    boxes = [(10, 10, 20, 20)]

    tracks, per_det = update_tracks_for_frame(boxes, tracks, iou_threshold=0.6)
    assert per_det[0].stable_frames == 1

    tracks, per_det = update_tracks_for_frame(boxes, tracks, iou_threshold=0.6)
    assert per_det[0].stable_frames == 2

    tracks, per_det = update_tracks_for_frame(boxes, tracks, iou_threshold=0.6)
    assert per_det[0].stable_frames == 3


def test_should_classify_track_only_after_min_frames_and_once():
    t = StableTrack(track_id=0, box=(0, 0, 10, 10), stable_frames=2, classified=False)
    assert should_classify_track(t, min_stable_frames=3) is False

    t.stable_frames = 3
    assert should_classify_track(t, min_stable_frames=3) is True

    t.classified = True
    assert should_classify_track(t, min_stable_frames=3) is False


def test_should_run_bird_classification_for_detection_after_consistent_iou_over_n_frames():
    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=3)

    # Same box across frames => stable_frames increments.
    for _frame in range(2):
        tracker.update_frame([DummyDet((10, 10, 20, 20))])
        assert should_run_bird_classification_for_detection(0, tracker=tracker) is False

    tracker.update_frame([DummyDet((10, 10, 20, 20))])
    assert should_run_bird_classification_for_detection(0, tracker=tracker) is True


def test_should_not_classify_if_track_breaks_iou_consistency():
    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=3)

    tracker.update_frame([DummyDet((10, 10, 20, 20))])
    tracker.update_frame([DummyDet((10, 10, 20, 20))])

    # Big jump -> IoU ~ 0 => new track, stability resets.
    tracker.update_frame([DummyDet((200, 200, 20, 20))])
    assert should_run_bird_classification_for_detection(0, tracker=tracker) is False

    # Needs 3 frames again for the new location.
    tracker.update_frame([DummyDet((200, 200, 20, 20))])
    assert should_run_bird_classification_for_detection(0, tracker=tracker) is False

    tracker.update_frame([DummyDet((200, 200, 20, 20))])
    assert should_run_bird_classification_for_detection(0, tracker=tracker) is True


def test_two_tracks_dont_steal_each_other():
    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=2)

    # Frame 1: two separate detections => two tracks.
    tracker.update_frame([DummyDet((10, 10, 20, 20)), DummyDet((200, 200, 20, 20))])
    assert should_run_bird_classification_for_detection(0, tracker=tracker) is False
    assert should_run_bird_classification_for_detection(1, tracker=tracker) is False

    # Frame 2: same ordering. Both should be stable enough now.
    tracker.update_frame([DummyDet((10, 10, 20, 20)), DummyDet((200, 200, 20, 20))])
    assert should_run_bird_classification_for_detection(0, tracker=tracker) is True
    assert should_run_bird_classification_for_detection(1, tracker=tracker) is True


def test_tracker_mark_classified_prevents_reclassification():
    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=2)

    tracker.update_frame([DummyDet((10, 10, 20, 20))])
    tracker.update_frame([DummyDet((10, 10, 20, 20))])

    assert should_run_bird_classification_for_detection(0, tracker=tracker) is True

    track = tracker.track_for_detection_id(0)
    assert track is not None
    tracker.mark_classified(track.track_id)

    # Even if the object stays stable, it should not be reclassified.
    tracker.update_frame([DummyDet((10, 10, 20, 20))])
    assert should_run_bird_classification_for_detection(0, tracker=tracker) is False


def test_process_flow_like_loop_with_deterministic_frames():
    """End-to-end-ish for gating: only triggers classify once after N stable frames."""

    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=3)

    classify_calls = []

    def fake_classify():
        classify_calls.append(1)

    frames = [
        [DummyDet((10, 10, 20, 20))],
        [DummyDet((10, 10, 20, 20))],
        [DummyDet((10, 10, 20, 20))],
        [DummyDet((10, 10, 20, 20))],
    ]

    for detections in frames:
        tracker.update_frame(detections)
        if should_run_bird_classification_for_detection(0, tracker=tracker):
            # emulate classify and mark classified
            fake_classify()
            track = tracker.track_for_detection_id(0)
            assert track is not None
            tracker.mark_classified(track.track_id)

    assert len(classify_calls) == 1


def test_multi_detection_two_stable_tracks_with_spurious_and_order_changes():
    """More realistic scenario.

    We simulate multiple detections per frame:
      - Track A: a bird sitting at ~ (10,10,20,20)
      - Track B: a bird sitting at ~ (200,200,20,20)
      - Spurious detections: boxes that come and go and should *not* become stable.

    Important: detection ordering is per-frame and can change.
    The tracker must match by IoU, not by detection_id.

    Expectation:
      - After N=3 stable frames, *both* A and B become classifiable.
      - The spurious detections never persist long enough to reach stability.
    """

    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=3)

    # Frame 1
    # - A and B appear
    # - plus one spurious detection C
    tracker.update_frame(
        _boxes_for_frame(
            (10, 10, 20, 20),  # A
            (200, 200, 20, 20),  # B
            (50, 50, 10, 10),  # C (spurious)
        )
    )
    assert _stable_set_after_update(tracker) == set()

    # Frame 2
    # - Order changes: B comes first, then A
    # - spurious changes location (so it should not match itself with IoU>0.6)
    tracker.update_frame(
        _boxes_for_frame(
            (200, 200, 20, 20),  # B
            (10, 10, 20, 20),  # A
            (80, 60, 10, 10),  # C' (spurious moved)
        )
    )
    assert _stable_set_after_update(tracker) == set()

    # Frame 3
    # - Another order change: A, then a *new* spurious D, then B
    # At this point A and B have each matched across 3 frames, so you should
    # see exactly two detection_ids allowed to classify (but which IDs depends
    # on the per-frame ordering).
    tracker.update_frame(
        _boxes_for_frame(
            (10, 10, 20, 20),  # A
            (300, 10, 10, 10),  # D (spurious)
            (200, 200, 20, 20),  # B
        )
    )
    assert len(_stable_set_after_update(tracker)) == 2


def test_multi_detection_with_small_jitter_still_counts_as_stable():
    """Variant: stable tracks jitter slightly but should still be matched.

    We keep the jitter small enough to keep IoU > 0.6 frame-to-frame.
    """

    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=3)

    # A jitters by a couple of pixels.
    # B jitters by a couple of pixels.
    frames = [
        _boxes_for_frame((10, 10, 20, 20), (200, 200, 20, 20), (50, 50, 12, 12)),
        _boxes_for_frame((12, 11, 20, 20), (199, 202, 20, 20), (51, 49, 12, 12)),
        _boxes_for_frame((11, 12, 20, 20), (201, 201, 20, 20), (20, 70, 12, 12)),
    ]

    for idx, detections in enumerate(frames):
        tracker.update_frame(detections)
        stable_ids = _stable_set_after_update(tracker)
        if idx < 2:
            assert stable_ids == set()
        else:
            # After 3rd frame, both stable tracks (A and B) should be eligible.
            assert len(stable_ids) == 2


def test_unstable_track_never_reaches_stability_when_it_keeps_teleporting():
    """An unstable object: it keeps jumping so IoU never exceeds the threshold.

    Even after many frames, it should never reach N stable frames.
    """

    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=3)

    # We always include a consistent A so the tracker isn't trivially empty.
    # The unstable U jumps around each frame.
    frames = [
        _boxes_for_frame((10, 10, 20, 20), (300, 300, 10, 10)),
        _boxes_for_frame((10, 10, 20, 20), (10, 300, 10, 10)),
        _boxes_for_frame((10, 10, 20, 20), (300, 10, 10, 10)),
        _boxes_for_frame((10, 10, 20, 20), (150, 150, 10, 10)),
        _boxes_for_frame((10, 10, 20, 20), (0, 0, 10, 10)),
    ]

    eligible_counts = []
    for detections in frames:
        tracker.update_frame(detections)
        eligible_counts.append(len(_stable_set_after_update(tracker)))

    # A becomes eligible once (on the 3rd frame), but U should not.
    assert eligible_counts[0] == 0
    assert eligible_counts[1] == 0
    assert eligible_counts[2] == 1
    assert eligible_counts[3] == 1
    assert eligible_counts[4] == 1


def test_tracks_are_deleted_after_max_missing_frames_without_match():
    """Tracks should be garbage-collected if they disappear for too long.

    Setup:
      - max_missing_frames=2 means: if a track is not matched for 3rd consecutive frame,
        it should be removed.

    Steps:
      1) Create a track A.
      2) Provide 2 empty frames: A is still retained (it is missing but not over the limit).
      3) Provide a 3rd empty frame: A should be deleted.
    """

    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=3, max_missing_frames=2)

    # Frame 1: A is detected -> 1 active track.
    tracker.update_frame([DummyDet((10, 10, 20, 20))])
    assert tracker.track_count() == 1

    # Frame 2: no detections -> A is now missing for 1 frame.
    tracker.update_frame([])
    assert tracker.track_count() == 1

    # Frame 3: no detections -> A is now missing for 2 frames.
    tracker.update_frame([])
    assert tracker.track_count() == 1

    # Frame 4: no detections -> A is now missing for 3 frames (> max_missing_frames).
    tracker.update_frame([])
    assert tracker.track_count() == 0


def test_track_reappearing_after_deletion_is_treated_as_new_and_needs_stability_again():
    """If a track is deleted due to staleness, a later detection in the same spot should
    create a new track (and thus re-require N stable frames).
    """

    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=2, max_missing_frames=1)

    # Frame 1: A appears.
    tracker.update_frame([DummyDet((10, 10, 20, 20))])
    assert tracker.track_count() == 1
    assert _stable_set_after_update(tracker) == set()  # min_stable_frames=2, so not yet

    # Frame 2: A disappears (missing frame 1) -> still retained because max_missing_frames=1.
    tracker.update_frame([])
    assert tracker.track_count() == 1

    # Frame 3: A disappears again (missing frame 2) -> should be deleted.
    tracker.update_frame([])
    assert tracker.track_count() == 0

    # Frame 4: A reappears at the same position.
    # It must be treated as a new track, i.e. stability is reset.
    tracker.update_frame([DummyDet((10, 10, 20, 20))])
    assert tracker.track_count() == 1
    assert _stable_set_after_update(tracker) == set()

    # Frame 5: A is seen again -> now it reaches min_stable_frames=2.
    tracker.update_frame([DummyDet((10, 10, 20, 20))])
    assert _stable_set_after_update(tracker) == {0}


def test_update_frame_can_filter_to_only_bird_detections():
    """The tracker should only create/maintain tracks for detections that pass keep_detection.

    This matches the runtime behavior where we only want to track detections that the model
    labels as 'bird'.

    Steps:
      1) Provide a frame with one 'bird' and one 'cat'. Only the bird should become a track.
      2) Provide a second frame with the same boxes. Bird track stability increments.
      3) Provide a third frame with only the non-bird; bird track becomes missing but is retained
         (default max_missing_frames=0 means no deletion).
    """

    class DetWithCategory(DummyDet):
        def __init__(self, box, category: str):
            super().__init__(box)
            self.category = category

    tracker = StableDetectionTracker(iou_threshold=0.6, min_stable_frames=2)

    # Frame 1: both appear, but we only keep 'bird'.
    tracker.update_frame(
        [
            DetWithCategory((10, 10, 20, 20), "bird"),
            DetWithCategory((50, 50, 20, 20), "cat"),
        ],
        keep_detection=lambda d: getattr(d, "category") == "bird",
    )
    assert tracker.track_count() == 1

    # Frame 2: bird is still there; stability increments to 2 -> becomes eligible.
    tracker.update_frame(
        [
            DetWithCategory((10, 10, 20, 20), "bird"),
            DetWithCategory((50, 50, 20, 20), "cat"),
        ],
        keep_detection=lambda d: getattr(d, "category") == "bird",
    )
    assert tracker.track_count() == 1
    assert len(_stable_set_after_update(tracker)) == 1

    # Frame 3: only non-bird detections. Since we filter them out, the tracker receives
    # an empty detection list.
    tracker.update_frame(
        [DetWithCategory((50, 50, 20, 20), "cat")],
        keep_detection=lambda d: getattr(d, "category") == "bird",
    )
    assert tracker.track_count() == 1
