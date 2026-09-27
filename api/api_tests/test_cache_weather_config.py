"""TTL cache, CWFIS parsing, and every environment variable in api/.env.example."""

from __future__ import annotations

import importlib
from pathlib import Path

import httpx
import pytest
from api_tests.conftest import REAL_FETCH_JSON, REAL_HTTPX_CLIENT
from app.cache import TTLCache
from app.config import (
    CWFIS_WFS_URL,
    DEFAULT_SOURCES,
    FIRMS_BASE_URL,
    get_settings,
)
from app.geo import compass_label, haversine_km
from app.weather import parse_stations
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[2]


def test_ttl_cache_hit_miss_expiry_and_per_key(monkeypatch):
    cache = TTLCache()
    clock = {"now": 1_000.0}
    monkeypatch.setattr("app.cache.time.monotonic", lambda: clock["now"])

    assert cache.get("missing") is None
    cache.set("hotspots", {"n": 1}, 10)
    cache.set("weather", {"n": 2}, 5)
    assert cache.get("hotspots") == {"n": 1}
    assert cache.get("weather") == {"n": 2}

    clock["now"] = 1_006.0
    assert cache.get("weather") is None
    assert cache.get("hotspots") == {"n": 1}

    clock["now"] = 1_011.0
    assert cache.get("hotspots") is None

    payload = {"rows": [1]}
    cache.set("copy", payload, 30)
    payload["rows"].append(2)
    first = cache.get("copy")
    assert first == {"rows": [1]}
    first["rows"].append(3)
    assert cache.get("copy") == {"rows": [1]}

    cache.set("skipped", {"n": 1}, 0)
    assert cache.get("skipped") is None
    cache.clear()
    assert cache.get("copy") is None
    cache.set("copy", {"n": 1}, 30)
    assert cache.get("copy") == {"n": 1}


def test_hotspot_cache_ttl_zero_and_expiry(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "k")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT")
    monkeypatch.setenv("HOTSPOT_CACHE_SECONDS", "0")
    calls = {"n": 0}

    def fake(_url: str, _timeout: float) -> str:
        calls["n"] += 1
        return "latitude,longitude\n49.1,-123.1\n"

    monkeypatch.setattr("app.firms.fetch_text", fake)
    assert client.get("/hotspots").json()["cached"] is False
    assert client.get("/hotspots").json()["cached"] is False
    assert calls["n"] == 2

    from app.firms import hotspot_cache

    hotspot_cache.clear()
    clock = {"now": 500.0}
    monkeypatch.setattr("app.cache.time.monotonic", lambda: clock["now"])
    monkeypatch.setenv("HOTSPOT_CACHE_SECONDS", "10")
    assert client.get("/hotspots").json()["cached"] is False
    clock["now"] = 509.0
    assert client.get("/hotspots").json()["cached"] is True
    clock["now"] = 511.0
    assert client.get("/hotspots").json()["cached"] is False
    assert calls["n"] == 4


def test_weather_and_hotspot_caches_do_not_share_keys(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "k")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT")
    firms_calls = {"n": 0}

    def firms(_url: str, _timeout: float) -> str:
        firms_calls["n"] += 1
        return "latitude,longitude\n49.1,-123.1\n"

    def weather(_url, _params, _timeout):
        return {"features": []}

    monkeypatch.setattr("app.firms.fetch_text", firms)
    monkeypatch.setattr("app.weather.fetch_json", weather)
    assert client.get("/weather").status_code == 200
    assert client.get("/hotspots").json()["cached"] is False
    assert client.get("/hotspots").json()["cached"] is True
    assert firms_calls["n"] == 1


def _station(**overrides):
    base = {
        "wmo": "100001",
        "name": "TEST",
        "rep_date": "2026-09-26T12:00:00Z",
        "lat": 49.2,
        "lon": -123.1,
        "fwi": 12,
    }
    base.update(overrides)
    return {"type": "Feature", "properties": base}


def test_parse_stations_skips_bad_payloads_and_keeps_latest():
    payload = {
        "features": [
            "nope",
            {"type": "Feature", "properties": ["not-a-dict"]},
            _station(fwi="high"),
            _station(lat="nan", fwi=10),
            _station(lat=10, lon=10, fwi=30),
            _station(wmo="", name="   ", fwi=8),
            _station(wmo="", name="  Name   Only ", fwi=4, rep_date="2026-09-20T00:00:00Z"),
            _station(wmo="", name="Name Only", fwi=9, rep_date="2026-09-21T00:00:00Z"),
            _station(rep_date="2026-09-25T00:00:00Z", fwi=1, ws="inf", temp=""),
            _station(rep_date="2026-09-27T00:00:00Z", fwi=6, rh="40"),
            {"type": "Feature"},
        ]
    }
    stations = parse_stations(payload)
    by_id = {item["id"]: item for item in stations}
    assert set(by_id) == {"100001", "Name Only"}
    assert by_id["100001"]["fwi"] == 6
    assert by_id["100001"]["relative_humidity"] == 40
    assert by_id["100001"]["wind_speed_kmh"] is None
    assert by_id["100001"]["temp_c"] is None
    assert by_id["Name Only"]["fwi"] == 9
    assert by_id["Name Only"]["name"] == "Name Only"
    assert parse_stations({}) == []
    assert parse_stations({"features": None}) == []


def test_weather_bad_payload_soft_fails_and_zero_ttl_does_not_stick(client, monkeypatch):
    monkeypatch.setenv("WEATHER_CACHE_SECONDS", "0")
    calls = {"n": 0}

    def fake(_url, _params, _timeout):
        calls["n"] += 1
        raise RuntimeError("cwfis down")

    monkeypatch.setattr("app.weather.fetch_json", fake)
    first = client.get("/weather")
    second = client.get("/weather")
    assert first.status_code == 200
    assert first.json()["available"] is False
    assert first.json()["stations"] == []
    assert second.json()["cached"] is False
    assert calls["n"] == 2


def test_weather_failure_is_cached_briefly_when_ttl_is_positive(client, monkeypatch):
    monkeypatch.setenv("WEATHER_CACHE_SECONDS", "30")
    calls = {"n": 0}

    def fake(_url, _params, _timeout):
        calls["n"] += 1
        raise RuntimeError("cwfis down")

    monkeypatch.setattr("app.weather.fetch_json", fake)
    assert client.get("/weather").json()["cached"] is False
    assert client.get("/weather").json()["cached"] is True
    assert calls["n"] == 1


def test_weather_http_error_and_non_json_are_soft_failures(client, monkeypatch):
    def raise_status(_url, _params, _timeout):
        request = httpx.Request("GET", "https://cwfis.example")
        response = httpx.Response(502, request=request, text="<html>bad</html>")
        raise httpx.HTTPStatusError("bad gateway", request=request, response=response)

    monkeypatch.setattr("app.weather.fetch_json", raise_status)
    failed = client.get("/weather")
    assert failed.status_code == 200
    assert failed.json()["available"] is False

    from app.weather import weather_cache

    weather_cache.clear()

    def not_json(_url, _params, _timeout):
        raise ValueError("CWFIS did not return GeoJSON.")

    monkeypatch.setattr("app.weather.fetch_json", not_json)
    again = client.get("/weather")
    assert again.json()["available"] is False
    assert again.json()["station_count"] == 0


def test_weather_empty_windows_are_available_with_no_stations(client, monkeypatch):
    calls = {"n": 0}

    def fake(_url, params, _timeout):
        calls["n"] += 1
        assert "prov='BC'" in params["CQL_FILTER"]
        assert params["typeName"] == "public:firewx_stns"
        return {"features": [_station(lat=10, lon=10, fwi=1)]}

    monkeypatch.setattr("app.weather.fetch_json", fake)
    body = client.get("/weather").json()
    assert calls["n"] == 2
    assert body["available"] is True
    assert body["station_count"] == 0
    assert "No recent" in body["detail"]


def test_weather_first_window_error_does_not_widen(client, monkeypatch):
    calls = {"n": 0}

    def fake(_url, _params, _timeout):
        calls["n"] += 1
        raise RuntimeError("timeout")

    monkeypatch.setattr("app.weather.fetch_json", fake)
    assert client.get("/weather").json()["available"] is False
    assert calls["n"] == 1


def test_fetch_json_rejects_non_objects_and_non_json():
    import app.weather as weather

    def html(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html></html>")

    def array(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2])

    def ok(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": []})

    def factory(handler):
        transport = httpx.MockTransport(handler)

        def client(*args, **kwargs):
            kwargs["transport"] = transport
            return REAL_HTTPX_CLIENT(*args, **kwargs)

        return client

    weather.httpx.Client = factory(html)
    try:
        with pytest.raises(weather.WeatherUnavailable):
            REAL_FETCH_JSON("https://cwfis.example/ows", {"service": "WFS"}, 5)
        weather.httpx.Client = factory(array)
        with pytest.raises(weather.WeatherUnavailable):
            REAL_FETCH_JSON("https://cwfis.example/ows", {"service": "WFS"}, 5)
        weather.httpx.Client = factory(ok)
        assert REAL_FETCH_JSON("https://cwfis.example/ows", {"service": "WFS"}, 5) == {
            "features": []
        }
    finally:
        weather.httpx.Client = REAL_HTTPX_CLIENT


def test_geo_helpers():
    assert haversine_km(49.0, -123.0, 49.0, -123.0) == 0
    assert 10 < haversine_km(53.0, -132.0, 53.1, -132.0) < 12
    assert compass_label(0) == "N"
    assert compass_label(22.5) == "NE"
    assert compass_label(45) == "NE"
    assert compass_label(90) == "E"
    assert compass_label(180) == "S"
    assert compass_label(270) == "W"
    assert compass_label(315) == "NW"
    assert compass_label(360) == "N"
    assert compass_label(-90) == "W"


def test_every_env_example_key_is_parsed(monkeypatch):
    example = (REPO / "api" / ".env.example").read_text()
    keys = []
    for line in example.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.append(stripped.split("=", 1)[0])
    assert keys == [
        "FIRMS_MAP_KEY",
        "XAI_API_KEY",
        "CORS_ORIGINS",
        "GROK_MODEL",
        "FIRMS_DAY_RANGE",
        "FIRMS_SOURCES",
        "HOTSPOT_CACHE_SECONDS",
        "WEATHER_CACHE_SECONDS",
        "HTTP_TIMEOUT_SECONDS",
    ]

    for name in keys:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("FIRMS_BASE_URL", raising=False)
    monkeypatch.delenv("CWFIS_WFS_URL", raising=False)
    defaults = get_settings()
    assert defaults.firms_map_key == ""
    assert defaults.xai_api_key == ""
    assert defaults.cors_origins == ("http://localhost:3000", "http://127.0.0.1:3000")
    assert defaults.grok_model == "grok-3"
    assert defaults.firms_day_range == 2
    assert defaults.firms_sources == DEFAULT_SOURCES
    assert defaults.hotspot_cache_seconds == 600
    assert defaults.weather_cache_seconds == 1800
    assert defaults.request_timeout == 30
    assert defaults.firms_base_url == FIRMS_BASE_URL
    assert defaults.cwfis_wfs_url == CWFIS_WFS_URL

    monkeypatch.setenv("FIRMS_MAP_KEY", "  firms-key  ")
    monkeypatch.setenv("XAI_API_KEY", "  xai-key ")
    monkeypatch.setenv("CORS_ORIGINS", " https://a.example , https://b.example ")
    monkeypatch.setenv("GROK_MODEL", "grok-3-mini")
    monkeypatch.setenv("FIRMS_DAY_RANGE", "4")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_NOAA20_NRT, MODIS_NRT")
    monkeypatch.setenv("HOTSPOT_CACHE_SECONDS", "15")
    monkeypatch.setenv("WEATHER_CACHE_SECONDS", "45")
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("FIRMS_BASE_URL", "https://firms.example/")
    monkeypatch.setenv("CWFIS_WFS_URL", "https://cwfis.example/ows")
    settings = get_settings()
    assert settings.firms_map_key == "firms-key"
    assert settings.xai_api_key == "xai-key"
    assert settings.cors_origins == ("https://a.example", "https://b.example")
    assert settings.grok_model == "grok-3-mini"
    assert settings.firms_day_range == 4
    assert settings.firms_sources == ("VIIRS_NOAA20_NRT",)
    assert settings.hotspot_cache_seconds == 15
    assert settings.weather_cache_seconds == 45
    assert settings.request_timeout == 12
    assert settings.firms_base_url == "https://firms.example"
    assert settings.cwfis_wfs_url == "https://cwfis.example/ows"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 1),
        ("1", 1),
        ("5", 5),
        ("6", 5),
        ("-3", 1),
        ("1.5", 2),
        ("", 2),
        (" 3 ", 3),
    ],
)
def test_firms_day_range_bounds(monkeypatch, raw, expected):
    monkeypatch.setenv("FIRMS_DAY_RANGE", raw)
    assert get_settings().firms_day_range == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", 30),
        ("4", 5),
        ("5", 5),
        ("120", 120),
        ("121", 120),
        ("abc", 30),
    ],
)
def test_http_timeout_bounds(monkeypatch, raw, expected):
    if raw == "":
        monkeypatch.delenv("HTTP_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", raw)
    assert get_settings().request_timeout == expected


def test_invalid_model_and_origins_and_cache_bounds(monkeypatch):
    monkeypatch.setenv("GROK_MODEL", "grok 3; rm")
    monkeypatch.setenv("CORS_ORIGINS", " , ")
    monkeypatch.setenv("HOTSPOT_CACHE_SECONDS", "99999")
    monkeypatch.setenv("WEATHER_CACHE_SECONDS", "-1")
    monkeypatch.setenv("CWFIS_WFS_URL", "   ")
    settings = get_settings()
    assert settings.grok_model == "grok-3"
    assert settings.cors_origins == ("http://localhost:3000",)
    assert settings.hotspot_cache_seconds == 3600
    assert settings.weather_cache_seconds == 0
    assert settings.cwfis_wfs_url == CWFIS_WFS_URL

    monkeypatch.setenv("CORS_ORIGINS", "*")
    assert get_settings().cors_origins == ("*",)


def _reload_app(monkeypatch: pytest.MonkeyPatch, origins: str | None) -> TestClient:
    if origins is None:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("CORS_ORIGINS", origins)
    import app.main as main

    importlib.reload(main)
    return TestClient(main.app)


def test_cors_default_and_configured_origins(monkeypatch):
    client = _reload_app(monkeypatch, None)
    allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    denied = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in denied.headers
    preflight = client.options(
        "/hotspots",
        headers={
            "Origin": "http://127.0.0.1:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"
    assert "GET" in preflight.headers["access-control-allow-methods"]

    custom = _reload_app(monkeypatch, "https://map.example, https://admin.example")
    ok = custom.get("/health", headers={"Origin": "https://admin.example"})
    assert ok.headers["access-control-allow-origin"] == "https://admin.example"
    blocked = custom.get("/health", headers={"Origin": "https://map.example.evil"})
    assert "access-control-allow-origin" not in blocked.headers

    wildcard = _reload_app(monkeypatch, "*")
    star = wildcard.get("/health", headers={"Origin": "https://anywhere.example"})
    assert star.headers["access-control-allow-origin"] == "*"
    assert star.headers.get("access-control-allow-credentials") in (None, "false")
