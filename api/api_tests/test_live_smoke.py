"""Opt-in network checks. The default suite never runs these."""

from __future__ import annotations

import os

import httpx
import pytest
from api_tests.conftest import REAL_FETCH_JSON, REAL_HTTPX_CLIENT
from app.config import CWFIS_WFS_URL

pytestmark = pytest.mark.live

_LIVE = os.environ.get("WILDFIRE_LIVE_SMOKE") == "1"
_FIRMS_KEY = os.environ.get("FIRMS_MAP_KEY", "").strip()


@pytest.mark.skipif(not _LIVE, reason="set WILDFIRE_LIVE_SMOKE=1 to call CWFIS")
def test_cwfis_wfs_smoke(monkeypatch):
    import app.weather as weather

    monkeypatch.setattr(httpx, "Client", REAL_HTTPX_CLIENT)
    monkeypatch.setattr(weather.httpx, "Client", REAL_HTTPX_CLIENT)
    payload = REAL_FETCH_JSON(
        CWFIS_WFS_URL,
        {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": "public:firewx_stns",
            "outputFormat": "application/json",
            "count": "1",
        },
        30,
    )
    assert isinstance(payload, dict)
    assert "features" in payload


@pytest.mark.skipif(
    not (_LIVE and _FIRMS_KEY),
    reason="set WILDFIRE_LIVE_SMOKE=1 and FIRMS_MAP_KEY to call NASA FIRMS",
)
def test_firms_area_csv_smoke(monkeypatch):
    """One VIIRS area request. The key is not asserted into the failure text."""
    monkeypatch.setattr(httpx, "Client", REAL_HTTPX_CLIENT)
    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{_FIRMS_KEY}/VIIRS_SNPP_NRT/-139,48,-114,60/1"
    )
    with REAL_HTTPX_CLIENT(timeout=45) as client:
        response = client.get(url, headers={"Accept": "text/csv"})
    assert response.status_code < 500
    # Do not echo the body: an error page can repeat the request URL.
