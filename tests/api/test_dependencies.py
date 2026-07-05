"""Unit tests for the FastAPI dependency providers in ``api.dependencies``.

The rest of the API suite overrides these providers via ``dependency_overrides``,
so the real provider bodies (env-var resolution + the read-only session factory)
are only exercised here.
"""

from pathlib import Path

from sqlmodel import Session

from birdscanner.api import dependencies
from birdscanner.db.database import init_db, make_engine


class TestGetImageDir:
    def test_defaults_when_env_unset(self, monkeypatch):
        """Without ``IMAGE_DIR`` set, the documented default path is returned."""
        monkeypatch.delenv("IMAGE_DIR", raising=False)
        assert dependencies.get_image_dir() == Path(
            "/home/stefan/Pictures/bird_detections"
        )

    def test_reads_env_var(self, monkeypatch, tmp_path):
        """``IMAGE_DIR`` overrides the default."""
        monkeypatch.setenv("IMAGE_DIR", str(tmp_path))
        assert dependencies.get_image_dir() == Path(str(tmp_path))


class TestGetReferenceDir:
    def test_defaults_to_repo_relative_dir(self, monkeypatch):
        """Without ``SPECIES_REFERENCE_DIR`` set, the repo-relative default is used."""
        monkeypatch.delenv("SPECIES_REFERENCE_DIR", raising=False)
        result = dependencies.get_reference_dir()
        assert result.name == "species_reference"
        assert result.parent.name == "assets"

    def test_reads_env_var(self, monkeypatch, tmp_path):
        """``SPECIES_REFERENCE_DIR`` overrides the default."""
        monkeypatch.setenv("SPECIES_REFERENCE_DIR", str(tmp_path))
        assert dependencies.get_reference_dir() == Path(str(tmp_path))


class TestGetSession:
    def test_yields_open_read_only_session(self, monkeypatch, tmp_path):
        """get_session lazily builds a read-only engine and yields a usable Session."""
        db_path = tmp_path / "detections.db"
        # Create the schema up-front: the read-only engine cannot create the file.
        init_db(make_engine(str(db_path)))

        monkeypatch.setenv("DB_PATH", str(db_path))
        # Reset the module-level singletons so the provider rebuilds against our DB.
        monkeypatch.setattr(dependencies, "_engine", None)
        monkeypatch.setattr(dependencies, "_session_factory", None)

        gen = dependencies.get_session()
        session = next(gen)
        try:
            assert isinstance(session, Session)
            # The engine is cached after first use.
            assert dependencies._engine is not None
            assert dependencies._session_factory is not None
        finally:
            gen.close()
