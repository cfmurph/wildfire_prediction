"""API tests import the FastAPI app without installing the research stack."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.cache import clear_all


def _blocked(*_args, **_kwargs):
    raise AssertionError("unexpected outbound request")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    clear_all()  # each test sees its own upstream responses
    monkeypatch.setattr(httpx, "AsyncClient", _blocked)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
