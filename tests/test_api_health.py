"""Health endpoint must work without any API call."""
from __future__ import annotations

from fastapi.testclient import TestClient

from mosaicmind.api.main import app


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["heavy_model"]
    assert body["text_embed_model"]
