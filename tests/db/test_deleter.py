"""Tests for ``db.deleter.delete_detection``.

Uses the shared in-memory ``session_factory`` + ``image_dir`` + ``detection_factory``
fixtures (top-level conftest), so no real detector or filesystem layout is needed.
"""

from birdscanner.db.deleter import delete_detection
from birdscanner.db.models import DetectionRecord


def test_delete_removes_row_and_files(session_factory, image_dir, detection_factory):
    """A successful delete removes the DB row and the image/thumbnail/video files."""
    record = detection_factory(species="Robin", track_id=1, with_video=True)

    assert delete_detection(session_factory, image_dir, record.id) is True

    with session_factory() as session:
        assert session.get(DetectionRecord, record.id) is None
    assert not (image_dir / "Robin/img_1.jpg").exists()
    assert not (image_dir / "Robin/img_1_thumb.jpg").exists()
    assert not (image_dir / "Robin/img_1.mp4").exists()


def test_delete_missing_record_returns_false(session_factory, image_dir):
    """Deleting an id that does not exist returns False and touches nothing."""
    assert delete_detection(session_factory, image_dir, 99999) is False


def test_delete_succeeds_when_image_files_already_gone(
    session_factory, image_dir, detection_factory
):
    """A missing image file does not block the row delete (best-effort unlink)."""
    record = detection_factory(species="Robin", track_id=1, with_video=True)
    # Remove the files out from under the deleter; the row must still be deleted.
    (image_dir / "Robin/img_1.jpg").unlink()
    (image_dir / "Robin/img_1_thumb.jpg").unlink()

    assert delete_detection(session_factory, image_dir, record.id) is True
    with session_factory() as session:
        assert session.get(DetectionRecord, record.id) is None
