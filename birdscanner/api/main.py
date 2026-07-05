"""FastAPI application for the BirdScanner backend.

Mounts the API routers and serves the pre-built React frontend from
``frontend/dist/`` at the root path.  When the dist directory is absent (e.g.
during development or testing) the static-file mount is skipped so the API
remains fully functional without a frontend build.

Environment variables:
    DB_PATH: Path to the SQLite database file (default: ``detections.db``).
    IMAGE_DIR: Root directory containing saved bird images
               (default: ``/home/stefan/Pictures/bird_detections``).
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from birdscanner.api.routers import (
    camera,
    detections,
    images,
    network,
    reference,
    species,
    system,
)

# The frontend build lives at the repo root (``<repo>/frontend/dist``), a sibling
# of the ``birdscanner`` package. This file is at ``<repo>/birdscanner/api/main.py``,
# so we walk up three levels (api -> birdscanner -> repo). ``parents[2]`` is used
# rather than ``.parent.parent`` — the latter stopped at ``<repo>/birdscanner`` after
# the codebase was unified into the ``birdscanner`` package (main.py moved a level
# deeper), so the mount silently never happened and ``/`` returned 404
# (``{"detail":"Not Found"}``). In Docker the api runs from ``/app`` with the build
# copied to ``/app/frontend/dist`` (see Dockerfile.api), which this also resolves.
_FRONTEND_DIST = Path(__file__).parents[2] / "frontend" / "dist"


class SPAStaticFiles(StaticFiles):
    """Static-file server that falls back to ``index.html`` for SPA routes.

    The React frontend uses client-side routing (``BrowserRouter``), so deep
    links such as ``/history`` have no matching file on disk.  A plain
    ``StaticFiles`` mount returns 404 (``{"detail": "Not Found"}``) for those
    paths, which breaks page reloads and shared links.  Overriding
    ``get_response`` to serve ``index.html`` whenever a path does not resolve
    to a real file lets the browser load the SPA, which then renders the
    correct route client-side.  API routes are unaffected because they are
    registered before this mount and take precedence.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve the requested file, or ``index.html`` when it is missing.

        Args:
            path: The request path relative to the mounted directory.
            scope: The ASGI connection scope for the request.

        Returns:
            The static-file response, falling back to ``index.html`` on a 404
            so client-side routes resolve.
        """
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


app = FastAPI(
    title="BirdScanner API",
    description="REST API for the BirdScanner real-time bird detection system.",
    version="2.0.0",
)

app.include_router(detections.router)
app.include_router(images.router)
app.include_router(system.router)
app.include_router(network.router)
app.include_router(species.router)
app.include_router(reference.router)
app.include_router(camera.router)

# Serve the React frontend if the build output exists.  SPAStaticFiles falls
# back to index.html so client-side routes (e.g. /history) load on direct
# navigation or refresh instead of returning a 404.
if _FRONTEND_DIST.is_dir():
    app.mount(
        "/", SPAStaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend"
    )
