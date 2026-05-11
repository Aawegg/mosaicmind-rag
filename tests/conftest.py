"""Pytest fixtures: ensure no real API calls are made when keys aren't present."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("mosaic-test-data")
    os.environ.setdefault("MOSAIC_DATA_DIR", str(root))
    os.environ.setdefault("MOSAIC_UPLOAD_DIR", str(root / "uploads"))
    os.environ.setdefault("MOSAIC_INDEX_DIR", str(root / "index"))
    os.environ.setdefault("MOSAIC_CHROMA_DIR", str(root / "chroma"))
    os.environ.setdefault("MLFLOW_TRACKING_URI", str(root / "mlruns"))
    os.environ.setdefault("GOOGLE_API_KEY", "dummy")
    os.environ.setdefault("GROQ_API_KEY", "dummy")
    Path(root).mkdir(parents=True, exist_ok=True)
    yield
