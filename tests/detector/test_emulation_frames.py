"""Tests for the emulated camera's frame sources."""

from pathlib import Path

import numpy as np
import pytest

from dev.emulation.frames import TestImagesSource, VideoSource

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_IMAGES = sorted((_REPO_ROOT / "tests" / "_test_images").glob("*.jpg"))


def test_test_images_source_cycles_frames(tmp_path):
    """TestImagesSource decodes images and serves them round-robin as RGB."""
    from PIL import Image

    paths = []
    for index, colour in enumerate([(200, 0, 0), (0, 200, 0)]):
        path = tmp_path / f"img_{index}.png"
        Image.new("RGB", (8, 6), colour).save(path)
        paths.append(path)

    source = TestImagesSource(paths)
    first = source.next_frame()
    second = source.next_frame()
    third = source.next_frame()  # wraps back to the first
    assert first is not None and second is not None and third is not None

    assert first.shape == (6, 8, 3)
    assert first[0, 0, 0] == 200 and first[0, 0, 1] == 0  # RGB, red image
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, third)


def test_test_images_source_rejects_empty():
    """An empty image list is a programming error."""
    with pytest.raises(ValueError):
        TestImagesSource([])


@pytest.mark.skipif(not _TEST_IMAGES, reason="bundled test images absent")
def test_test_images_source_reads_bundled_jpegs():
    """The bundled tests/_test_images JPEGs load as RGB frames."""
    source = TestImagesSource(_TEST_IMAGES)
    frame = source.next_frame()
    assert frame is not None
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


def _write_tiny_video(path: Path, frames: int = 3) -> bool:
    """Write a tiny mp4; return False if no usable encoder is available."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (16, 12))
    if not writer.isOpened():
        return False
    for value in range(frames):
        writer.write(np.full((12, 16, 3), 40 + value * 20, dtype=np.uint8))
    writer.release()
    return path.exists() and path.stat().st_size > 0


def test_video_source_reads_and_loops(tmp_path):
    """VideoSource reads frames as RGB and loops back at end-of-stream."""
    video_path = tmp_path / "clip.mp4"
    if not _write_tiny_video(video_path):
        pytest.skip("no mp4 encoder available in this OpenCV build")

    source = VideoSource(str(video_path), loop=True)
    try:
        frames = [source.next_frame() for _ in range(5)]
    finally:
        source.release()

    assert all(f is not None for f in frames)
    first = frames[0]
    assert first is not None
    assert first.ndim == 3 and first.shape[2] == 3


def test_video_source_rejects_missing_file(tmp_path):
    """Opening a non-existent video raises ValueError."""
    with pytest.raises(ValueError):
        VideoSource(str(tmp_path / "does_not_exist.mp4"))
