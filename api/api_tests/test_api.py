"""Tests for `api.main:app`, the API Docker and Railway start."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import httpx
import pytest
from api.services.predictor import risk_grid, spread_forecast

REPO = Path(__file__).resolve().parents[2]


def test_deploy_files_parse_and_the_image_can_see_requirements():
    dockerfile = (REPO / "api" / "Dockerfile").read_text()
    railway = tomllib.loads((REPO / "railway.toml").read_text())
    copied = [
        line.split()[1]
        for line in dockerfile.splitlines()
        if line.startswith("COPY ") and "requirements.txt" in line
    ]
    assert copied
    for source in copied:
        assert (REPO / source).is_file(), source
    assert railway["deploy"]["healthcheckPath"] == "/health"
    assert "uvicorn" in railway["deploy"]["startCommand"]
    assert "CMD" in dockerfile


def test_root_names_the_service(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "wildfire-prediction-api"
    assert body["status"] == "ok"
    assert body["docs"] == "/docs"


def test_health_does_not_leak_keys(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "super-secret-firms")
    response = client.get("/health")
    assert response.json() == {"status": "ok"}
    assert "super-secret" not in response.text


class _AsyncClient:
    def __init__(self, responder):
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, url, **_kwargs):
        return self._responder(url)


def _http_response(url: str, *, text: str = "", json_data: object | None = None, status: int = 200):
    request = httpx.Request("GET", url)
    if json_data is not None:
        return httpx.Response(status, json=json_data, request=request)
    return httpx.Response(status, text=text, request=request)


def _install_async(monkeypatch, responder):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _AsyncClient(responder))


def test_hotspots_without_key_are_empty(client, monkeypatch):
    def boom(_url):
        raise AssertionError("upstream called without a key")

    _install_async(monkeypatch, boom)
    response = client.get("/api/v1/fires/hotspots")
    assert response.status_code == 200
    assert response.json() == {"type": "FeatureCollection", "features": []}


@pytest.mark.parametrize("days", [0, 11])
def test_hotspot_day_query_is_bounded(client, days):
    response = client.get(f"/api/v1/fires/hotspots?days={days}")
    assert response.status_code == 422


def test_hotspots_skip_bad_rows_and_do_not_log_the_key(client, monkeypatch, caplog):
    monkeypatch.setenv("FIRMS_MAP_KEY", "super-secret-firms")
    csv = (
        "latitude,longitude,bright_ti4,bright_ti5,frp,confidence,acq_date,acq_time,satellite,daynight\n"
        "49.1,-123.1,300,290,12,high,2026-09-26,1900,N,D\n"
        "nope,-123.0,1,1,bad,low,2026-09-26,1900,N,D\n"
        "10,10,1,1,5,low,2026-09-26,1900,N,D\n"
        "53.0,-132.0,310,290,8,nominal,2026-09-26,1800,N20,N\n"
    )
    seen: list[str] = []

    def respond(url: str):
        seen.append(url)
        return _http_response(url, text=csv)

    _install_async(monkeypatch, respond)
    response = client.get("/api/v1/fires/hotspots?days=2")
    assert response.status_code == 200
    features = response.json()["features"]
    assert len(features) == 4  # 2 valid BC rows from each of the 2 NRT sources
    assert features[0]["geometry"]["coordinates"] == [-123.1, 49.1]
    assert seen[0].endswith("/VIIRS_NOAA21_NRT/-139,48,-114,60/2")
    assert seen[1].endswith("/VIIRS_NOAA20_NRT/-139,48,-114,60/2")
    assert "super-secret-firms" in seen[0]
    assert "super-secret-firms" not in response.text
    assert "super-secret-firms" not in caplog.text


def test_hotspot_upstream_error_is_empty_and_hides_the_key(client, monkeypatch, caplog):
    monkeypatch.setenv("FIRMS_MAP_KEY", "super-secret-firms")

    def respond(url: str):
        request = httpx.Request("GET", url)
        response = httpx.Response(500, request=request, text="nope")
        raise httpx.HTTPStatusError("nope", request=request, response=response)

    _install_async(monkeypatch, respond)
    response = client.get("/api/v1/fires/hotspots")
    assert response.status_code == 200
    assert response.json()["features"] == []
    assert "super-secret-firms" not in caplog.text
    assert "super-secret-firms" not in response.text


def test_active_fires_keep_valid_points_when_one_feature_is_bad(client, monkeypatch):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"FIRE_NUMBER": "N1"}, "geometry": None},
            {
                "type": "Feature",
                "properties": {
                    "FIRE_NUMBER": "N2", "FIRE_NAME": "Test", "CURRENT_SIZE": "12.5",
                    "IGNITION_DATE": 1784243916000, "GEOGRAPHIC_DESCRIPTION": "6.4 km S of Kluskoil Lake",
                    "FIRE_OF_NOTE_IND": "Y",
                },
                "geometry": {"type": "Point", "coordinates": [-123.1, 49.2]},
            },
            {"type": "Feature", "properties": [], "geometry": {"type": "Polygon", "coordinates": []}},
        ],
    }

    def respond(url: str):
        return _http_response(url, json_data=payload)

    _install_async(monkeypatch, respond)
    response = client.get("/api/v1/fires/active")
    assert response.status_code == 200
    features = response.json()["features"]
    assert len(features) == 1
    assert features[0]["properties"]["fire_number"] == "N2"
    assert features[0]["properties"]["size_ha"] == 12.5
    assert features[0]["properties"]["lat"] == 49.2
    assert features[0]["properties"]["ignition_date"] == "2026-07-16"
    assert features[0]["properties"]["location"] == "6.4 km S of Kluskoil Lake"
    assert features[0]["properties"]["fire_of_note"] is True


def test_fire_upstream_failure_is_an_error_not_zero_fires(client, monkeypatch):
    def respond(_url: str):
        raise httpx.ConnectError("down")

    _install_async(monkeypatch, respond)
    assert client.get("/api/v1/fires/active").status_code == 502
    assert client.get("/api/v1/fires/history?year=2018").status_code == 502


def test_history_tolerates_a_bad_size(client, monkeypatch):
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"FIRE_NUMBER": "H1", "FIRE_YEAR": "nope", "FIRE_SIZE_HECTARES": "lots"},
                "geometry": {"type": "Polygon", "coordinates": []},
            }
        ],
    }

    def respond(url: str):
        assert "FIRE_YEAR%3D2018" in url or "FIRE_YEAR=2018" in url
        return _http_response(url, json_data=payload)

    _install_async(monkeypatch, respond)
    response = client.get("/api/v1/fires/history?year=2018")
    assert response.status_code == 200
    props = response.json()["features"][0]["properties"]
    assert props["fire_number"] == "H1"
    assert props["fire_year"] == 2018
    assert props["size_ha"] == 0


def test_history_year_bounds(client):
    assert client.get("/api/v1/fires/history?year=2005").status_code == 422
    from datetime import date
    assert client.get(f"/api/v1/fires/history?year={date.today().year + 1}").status_code == 422


def test_history_falls_back_to_the_current_season_layer(client, monkeypatch):
    perimeter = {
        "type": "Feature",
        "properties": {"FIRE_NUMBER": "G1", "FIRE_YEAR": 2026, "FIRE_SIZE_HECTARES": 3, "FIRE_STATUS": "Out"},
        "geometry": {"type": "Polygon", "coordinates": [[[-123, 50], [-123, 50.1], [-122.9, 50], [-123, 50]]]},
    }

    def respond(url: str):
        current = "PROT_CURRENT_FIRE_POLYS_SP" in url
        return _http_response(url, json_data={"type": "FeatureCollection", "features": [perimeter] if current else []})

    _install_async(monkeypatch, respond)
    props = client.get("/api/v1/fires/history?year=2026").json()["features"][0]["properties"]
    assert props["fire_number"] == "G1"
    assert props["fire_status"] == "Out"


def test_predict_and_risk_are_local_heuristics(client):
    spread = client.get(
        "/api/v1/predict/spread",
        params={"lat": 53.0, "lon": -124.0, "wind_dir": 270, "wind_speed": 5, "fwi": 20, "isi": 8},
    )
    assert spread.status_code == 200
    body = spread.json()
    percentiles = {feature["properties"]["percentile"] for feature in body["features"]}
    assert percentiles == {"p25", "p50", "p75"}
    assert all(feature["properties"]["area_ha"] > 0 for feature in body["features"])
    direct = spread_forecast(lat=53.0, lon=-124.0)
    assert {feature["properties"]["percentile"] for feature in direct["features"]} == {
        "p25",
        "p50",
        "p75",
    }

    risk = client.get("/api/v1/risk/map")
    assert risk.status_code == 200
    grid = risk.json()
    assert grid["metadata"]["model"] == "climatology_v1"
    assert grid["features"]
    assert all(0 < item["properties"]["burn_probability"] <= 1 for item in grid["features"])
    assert risk_grid(n_lat=5, n_lon=5)["features"]


def test_weather_does_not_invent_stations_when_upstream_fails(client, monkeypatch):
    def respond(_url: str):
        raise httpx.ConnectError("cwfis down")

    _install_async(monkeypatch, respond)
    response = client.get("/api/v1/weather/fwi")
    assert response.status_code == 200
    assert response.json()["features"] == []


def test_weather_uses_live_rows_when_present(client, monkeypatch):
    def respond(url: str):
        return _http_response(
            url,
            json_data={"features": [
                {"properties": {"lat": 49.2, "lon": -123.1, "fwi": 18, "aes": "1", "name": "LIVE  "}},
                {"properties": {"lat": 10, "lon": 10, "fwi": 40, "aes": "2", "name": "OUT"}},
                {"properties": {"lat": 50.1, "lon": -120.0, "fwi": None, "aes": "3", "name": "NO FWI"}},
            ]},
        )

    _install_async(monkeypatch, respond)
    features = client.get("/api/v1/weather/fwi").json()["features"]
    assert len(features) == 1
    assert features[0]["properties"]["station_name"] == "LIVE"
    assert features[0]["properties"]["FWI"] == 18
    assert "is_synthetic" not in features[0]["properties"]


def _station_feed(url: str):
    return _http_response(
        url,
        json_data={"features": [
            {"properties": {"lat": 53.1, "lon": -124.0, "fwi": 25, "isi": 9, "ws": 18, "wdir": 225, "name": "NEAR"}},
        ]},
    )


def test_nearest_station_is_within_range_or_null(client, monkeypatch):
    _install_async(monkeypatch, _station_feed)
    assert client.get("/api/v1/weather/nearest?lat=53.0&lon=-124.0").json()["station_name"] == "NEAR"
    assert client.get("/api/v1/weather/nearest?lat=58.0&lon=-124.0").json() is None


def test_cors_allows_only_configured_origins(client):
    allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
    blocked = client.get("/health", headers={"Origin": "https://evil.example"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-origin" not in blocked.headers
