"""API tests import the FastAPI app without installing the research stack."""

from __future__ import annotations

import httpx
import openai
import pytest
from app.firms import fetch_text as real_fetch_text
from app.firms import hotspot_cache
from app.main import app
from app.weather import fetch_json as real_fetch_json
from app.weather import weather_cache
from fastapi.testclient import TestClient

REAL_FETCH_JSON = real_fetch_json
REAL_FETCH_TEXT = real_fetch_text
REAL_HTTPX_CLIENT = httpx.Client


def _blocked_async_client(*_args, **_kwargs):
    raise AssertionError("unexpected outbound HTTP client")


def _blocked_openai(*_args, **_kwargs):
    raise AssertionError("unexpected OpenAI client")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch):
    hotspot_cache.clear()
    weather_cache.clear()
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("WILDFIRE_LIVE_SMOKE", raising=False)
    monkeypatch.delenv("WILDFIRE_LIVE_GROK", raising=False)

    def blocked(*_args, **_kwargs):
        raise AssertionError("unexpected outbound request")

    monkeypatch.setattr("app.firms.fetch_text", blocked)
    monkeypatch.setattr("app.weather.fetch_json", blocked)
    monkeypatch.setattr(httpx, "AsyncClient", _blocked_async_client)
    monkeypatch.setattr(openai, "OpenAI", _blocked_openai)
    monkeypatch.setattr(openai, "AsyncOpenAI", _blocked_openai)
    yield
    hotspot_cache.clear()
    weather_cache.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
