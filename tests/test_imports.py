"""Smoke test: every public module imports cleanly without API calls."""
from __future__ import annotations


def test_package_imports():
    import mosaicmind  # noqa: F401
    from mosaicmind import config, llm  # noqa: F401
    from mosaicmind.agents import graph, prompts, tools  # noqa: F401
    from mosaicmind.api import main, schemas  # noqa: F401
    from mosaicmind.api.routes import eval as eval_route  # noqa: F401
    from mosaicmind.api.routes import health, ingest, query  # noqa: F401
    from mosaicmind.indexing import store  # noqa: F401
    from mosaicmind.ingestion import audio, base, image, pdf, router, video  # noqa: F401
    from mosaicmind.mlops import eval as ml_eval  # noqa: F401
    from mosaicmind.mlops import tracking  # noqa: F401


def test_settings_load():
    from mosaicmind.config import get_settings

    s = get_settings()
    assert s.heavy_model
    assert s.fast_model
    assert s.text_embed_model
    assert s.text_embed_dim > 0
