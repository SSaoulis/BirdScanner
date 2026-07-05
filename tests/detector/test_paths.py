"""Tests for package-anchored data-file path resolution + env overrides."""

from pathlib import Path

from birdscanner.detector import paths


def test_assets_dir_default(monkeypatch):
    """Without an override, the asset root is repo-relative (<repo>/assets)."""
    monkeypatch.delenv("ASSETS_DIR", raising=False)
    assert paths.assets_dir() == paths._REPO_ROOT / "assets"


def test_assets_dir_env_override(monkeypatch, tmp_path):
    """ASSETS_DIR overrides the asset root."""
    monkeypatch.setenv("ASSETS_DIR", str(tmp_path))
    assert paths.assets_dir() == Path(str(tmp_path))


def test_model_dir_default(monkeypatch):
    """MODEL_DIR defaults to ``<assets>/models``."""
    monkeypatch.delenv("MODEL_DIR", raising=False)
    monkeypatch.delenv("ASSETS_DIR", raising=False)
    assert paths.model_dir() == paths.assets_dir() / "models"


def test_model_dir_env_override(monkeypatch, tmp_path):
    """MODEL_DIR overrides the model directory independently of ASSETS_DIR."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert paths.model_dir() == Path(str(tmp_path))


def test_concrete_file_helpers(monkeypatch, tmp_path):
    """The concrete file helpers compose the resolved directories."""
    monkeypatch.setenv("ASSETS_DIR", str(tmp_path))
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert paths.coco_labels_path() == tmp_path / "labels" / "coco_labels.txt"
    assert (
        paths.class_to_idx_path()
        == tmp_path / "models" / "convnext_v2_tiny.onnx_class_to_idx.json"
    )
    assert (
        paths.classifier_model_path()
        == tmp_path / "models" / "convnext_v2_tiny_int8.onnx"
    )
