"""FastAPI application for the BirdScanner backend.

Mounts the four API routers and serves the pre-built React frontend from
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

from backend.routers import camera, detections, images, species, system

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

app = FastAPI(
    title="BirdScanner API",
    description="REST API for the BirdScanner real-time bird detection system.",
    version="2.0.0",
)

app.include_router(detections.router)
app.include_router(images.router)
app.include_router(system.router)
app.include_router(species.router)
app.include_router(camera.router)

# Serve the React frontend if the build output exists.
if _FRONTEND_DIST.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend"
    )
