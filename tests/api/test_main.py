"""Tests for the app factory's SPA static-file mount and dist-path resolution."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class TestSPAStaticFiles:
    """SPAStaticFiles must serve index.html for client-side routes.

    The React app uses BrowserRouter, so deep links like ``/history`` have no
    file on disk. The mount must fall back to ``index.html`` instead of 404 so
    the SPA loads and renders the route client-side.
    """

    @pytest.fixture()
    def spa_client(self, tmp_path: Path) -> TestClient:
        """App with SPAStaticFiles mounted over a temporary dist directory."""
        from fastapi import FastAPI

        from birdscanner.api.main import SPAStaticFiles

        (tmp_path / "index.html").write_text("<!doctype html><div id=root>")
        (tmp_path / "asset.js").write_text("console.log('asset');")

        app = FastAPI()
        app.mount(
            "/", SPAStaticFiles(directory=str(tmp_path), html=True), name="frontend"
        )
        return TestClient(app)

    def test_root_serves_index(self, spa_client):
        resp = spa_client.get("/")
        assert resp.status_code == 200
        assert "id=root" in resp.text

    def test_deep_link_falls_back_to_index(self, spa_client):
        resp = spa_client.get("/history")
        assert resp.status_code == 200
        assert "id=root" in resp.text

    def test_real_asset_still_served(self, spa_client):
        resp = spa_client.get("/asset.js")
        assert resp.status_code == 200
        assert "console.log" in resp.text

    def test_frontend_dist_resolves_to_repo_root(self):
        """``_FRONTEND_DIST`` must point at ``<repo>/frontend/dist``.

        The frontend build is a sibling of the ``birdscanner`` package, not nested
        inside it. When the codebase was unified into the ``birdscanner`` package,
        ``main.py`` moved a directory deeper but the path still walked only two
        parents, so it resolved to ``<repo>/birdscanner/frontend/dist`` (which never
        exists). The mount was silently skipped and ``/`` returned 404
        (``{"detail":"Not Found"}``). This pins the resolution so it cannot regress.
        """
        import birdscanner
        from birdscanner.api.main import _FRONTEND_DIST

        package_dir = Path(birdscanner.__file__).resolve().parent  # <repo>/birdscanner
        expected = package_dir.parent / "frontend" / "dist"  # <repo>/frontend/dist
        assert _FRONTEND_DIST == expected
