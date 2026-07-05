"""Fixtures shared across the FastAPI backend tests.

Builds ``TestClient``s with the database + image-directory dependencies pointed at
the in-memory fixtures from the global ``conftest.py``, seeds a small standard set
of detections, and provides a minimal ``httpx`` response stand-in for the routes
that proxy to the detector.
"""

from datetime import datetime, timezone
from typing import Any, Callable, Generator, List

import pytest
from fastapi.testclient import TestClient

from birdscanner.db.models import DetectionRecord


class FakeHttpxResponse:
    """Minimal ``httpx.Response`` stand-in for the detector-proxy routes.

    Covers the camera snapshot/crop proxies, the detection-delete proxy, and the
    speed-test route: exposes ``content``/``headers``/``status_code``/``text`` and a
    ``json()`` returning a preset body.
    """

    def __init__(
        self,
        content: bytes = b"",
        content_type: str = "image/jpeg",
        status_code: int = 200,
        json_body: Any = None,
    ) -> None:
        """Store the canned response attributes the routes read."""
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status_code
        self.text = content.decode("utf-8", "replace")
        self._json_body = json_body

    def raise_for_status(self) -> None:
        """No-op: the fakes represent already-successful upstream responses."""
        return None

    def json(self) -> Any:
        """Return the preset JSON body."""
        return self._json_body


@pytest.fixture()
def fake_httpx_response() -> type[FakeHttpxResponse]:
    """Expose :class:`FakeHttpxResponse` to tests that monkeypatch ``httpx``."""
    return FakeHttpxResponse


@pytest.fixture()
def make_client(
    session_factory, image_dir
) -> Generator[Callable[[], TestClient], None, None]:
    """Return a builder that yields a ``TestClient`` with DB + image deps overridden.

    The overrides are cleared on fixture teardown. Tests that need a specific data
    set seed it via ``detection_factory`` before calling the returned builder.
    """
    from birdscanner.api.main import app
    from birdscanner.api.dependencies import get_session, get_image_dir

    def _make() -> TestClient:
        def _override_session():
            with session_factory() as session:
                yield session

        app.dependency_overrides[get_session] = _override_session
        app.dependency_overrides[get_image_dir] = lambda: image_dir
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


@pytest.fixture()
def seeded_detections(detection_factory) -> List[DetectionRecord]:
    """Insert the standard three-detection set (2 Robin, 1 Sparrow, distinct days)."""
    return [
        detection_factory(
            species="Robin",
            confidence=0.95,
            track_id=1,
            ts=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        ),
        detection_factory(
            species="Robin",
            confidence=0.88,
            track_id=2,
            ts=datetime(2024, 6, 2, 12, 0, 0, tzinfo=timezone.utc),
        ),
        detection_factory(
            species="Sparrow",
            confidence=0.91,
            track_id=3,
            ts=datetime(2024, 6, 3, 12, 0, 0, tzinfo=timezone.utc),
        ),
    ]


@pytest.fixture()
def client(make_client, seeded_detections) -> TestClient:
    """A ``TestClient`` backed by the standard seeded detections."""
    return make_client()
