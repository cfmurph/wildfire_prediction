"""API tests import the FastAPI app without installing the research stack."""

from __future__ import annotations

import httpx
import openai
import pytest
from api.main import app
from fastapi.testclient import TestClient


def _blocked(*_args, **_kwargs):
    raise AssertionError("unexpected outbound request")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _blocked)
    monkeypatch.setattr(openai, "AsyncOpenAI", _blocked)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
