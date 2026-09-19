"""Web API: config discovery and a real prediction round-trip."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from webapp import server

    return TestClient(server.app)


def test_api_config_lists_tracks_and_samples(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    d = r.json()
    assert d["resolution"] > 0
    assert d["samples"] and d["tracks"]
    assert all("name" in t and "type" in t for t in d["tracks"])
    # every assay must be offered as a knockout target
    assert set(d["knockoutable"]) == {t["name"] for t in d["tracks"]}


def test_predict_rejects_tiny_region(client):
    r = client.post("/api/predict", json={"region": "chr2:20000000-20050000"})
    assert r.status_code == 400


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Hi-C prediction explorer" in r.content
