"""API tests import the FastAPI app without installing the research stack."""

from __future__ import annotations

import pytest
from app.firms import hotspot_cache
from app.main import app
from app.weather import weather_cache
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch):
    hotspot_cache.clear()
    weather_cache.clear()
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    def blocked(*_args, **_kwargs):
        raise AssertionError("unexpected outbound request")

    monkeypatch.setattr("app.firms.fetch_text", blocked)
    monkeypatch.setattr("app.weather.fetch_json", blocked)
    yield
    hotspot_cache.clear()
    weather_cache.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
