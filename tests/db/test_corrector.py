"""Tests for ``db.corrector.correct_detection_species``.

Uses the shared in-memory ``session_factory`` + ``image_dir`` + ``detection_factory``
fixtures (top-level conftest), so no real detector or filesystem layout is needed.
"""

import logging

from birdscanner.db.corrector import correct_detection_species
from birdscanner.db.models import DetectionRecord


def test_correct_moves_files_and_updates_row(
    session_factory, image_dir, detection_factory
):
    """A correction moves the still/thumbnail/clip and rewrites the row + paths."""
    record = detection_factory(species="Robin", track_id=1, with_video=True)

    result = correct_detection_species(session_factory, image_dir, record.id, "Sparrow")

    assert result is not None
    assert result["species"] == "Sparrow"
    assert result["corrected"] is True
    assert result["original_species"] == "Robin"
    assert result["image_path"] == "Sparrow/img_1.jpg"
    assert result["thumbnail_path"] == "Sparrow/img_1_thumb.jpg"
    assert result["video_path"] == "Sparrow/img_1.mp4"

    # Files moved to the corrected species folder.
    assert (image_dir / "Sparrow/img_1.jpg").exists()
    assert (image_dir / "Sparrow/img_1_thumb.jpg").exists()
    assert (image_dir / "Sparrow/img_1.mp4").exists()
    assert not (image_dir / "Robin/img_1.jpg").exists()

    # The DB row reflects the correction.
    with session_factory() as session:
        row = session.get(DetectionRecord, record.id)
        assert row.species == "Sparrow"
        assert row.corrected is True
        assert row.original_species == "Robin"
        assert row.image_path == "Sparrow/img_1.jpg"


def test_correct_keeps_original_confidence(
    session_factory, image_dir, detection_factory
):
    """The model's original confidence is preserved (only species changes)."""
    record = detection_factory(species="Robin", confidence=0.42, track_id=1)

    result = correct_detection_species(session_factory, image_dir, record.id, "Sparrow")

    assert result is not None
    assert result["confidence"] == 0.42


def test_correct_preserves_first_guess_across_repeat_corrections(
    session_factory, image_dir, detection_factory
):
    """``original_species`` keeps the model's *first* guess through re-corrections."""
    record = detection_factory(species="Robin", track_id=1)

    correct_detection_species(session_factory, image_dir, record.id, "Sparrow")
    result = correct_detection_species(session_factory, image_dir, record.id, "Wren")

    assert result is not None
    assert result["species"] == "Wren"
    assert result["original_species"] == "Robin"
    assert (image_dir / "Wren/img_1.jpg").exists()


def test_correct_missing_record_returns_none(session_factory, image_dir):
    """Correcting an id that does not exist returns None and touches nothing."""
    assert correct_detection_species(session_factory, image_dir, 99999, "Robin") is None


def test_correct_same_species_is_noop(session_factory, image_dir, detection_factory):
    """Correcting to the current species leaves the row unflagged and files in place."""
    record = detection_factory(species="Robin", track_id=1)

    result = correct_detection_species(session_factory, image_dir, record.id, "Robin")

    assert result is not None
    assert result["species"] == "Robin"
    assert not result["corrected"]
    assert result["original_species"] is None
    assert (image_dir / "Robin/img_1.jpg").exists()


def test_correct_succeeds_when_a_file_is_missing(
    session_factory, image_dir, detection_factory, caplog
):
    """A missing source file does not block the row update (best-effort move)."""
    record = detection_factory(species="Robin", track_id=1)
    # Remove the still out from under the corrector; the row must still update.
    (image_dir / "Robin/img_1.jpg").unlink()

    with caplog.at_level(logging.WARNING, logger="tracking"):
        result = correct_detection_species(
            session_factory, image_dir, record.id, "Sparrow"
        )

    assert result is not None
    assert result["species"] == "Sparrow"
    assert result["image_path"] == "Sparrow/img_1.jpg"
    # The thumbnail (which did exist) was still moved.
    assert (image_dir / "Sparrow/img_1_thumb.jpg").exists()
