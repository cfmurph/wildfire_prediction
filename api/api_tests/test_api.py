"""Tests for `api.main:app` and the API Docker image."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import psycopg
import pytest

from api.services import spread_store

REPO = Path(__file__).resolve().parents[2]


def test_deploy_files_parse_and_the_image_can_see_requirements():
    dockerfile = (REPO / "api" / "Dockerfile").read_text()
    copied = [
        line.split()[1]
        for line in dockerfile.splitlines()
        if line.startswith("COPY ") and "requirements.txt" in line
    ]
    assert copied
    for source in copied:
        assert (REPO / source).is_file(), source
    assert "CMD" in dockerfile


def test_root_names_the_service(client):
    response = client.get("/api/v1")
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


def test_fire_starts_follow_the_season_and_never_invent_data(client, monkeypatch, tmp_path):
    july = client.get("/api/v1/ignition/causes?month=7").json()
    january = client.get("/api/v1/ignition/causes?month=1").json()
    assert len(july["lightning"]["features"]) > 1000
    assert len(january["lightning"]["features"]) == 0
    assert july["unit"] == "fire starts per 1,000 km² in a typical month"
    assert all(f["properties"]["starts"] > 0 for f in july["human"]["features"])

    from api.routers import ignition

    monkeypatch.setattr(ignition, "FIRE_STARTS", tmp_path / "missing.json")
    ignition._fire_starts.cache_clear()
    try:
        assert client.get("/api/v1/ignition/causes?month=7").status_code == 503
    finally:
        ignition._fire_starts.cache_clear()


def test_spread_map_serves_the_latest_prediction_or_says_none_ran(client, monkeypatch, tmp_path):
    from api.routers import predict

    monkeypatch.setattr(predict, "SPREAD_MAP", tmp_path / "spread.geojson")
    assert client.get("/api/v1/predict/spread-map").status_code == 503

    (tmp_path / "spread.geojson").write_text(
        '{"type": "FeatureCollection", "features": [], "date": "2026-08-14", "issued": "2026-08-15T14:00+00:00"}'
    )
    response = client.get("/api/v1/predict/spread-map")
    assert response.headers["content-type"] == "application/geo+json"
    assert response.json()["date"] == "2026-08-14"

    os.utime(tmp_path / "spread.geojson", (0, 0))
    assert client.get("/api/v1/predict/spread-map").status_code == 503


@pytest.fixture
def spread_db(monkeypatch):
    """The test database, emptied. Never the real one: needs TEST_DATABASE_URL."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    with psycopg.connect(url) as conn:
        conn.execute("TRUNCATE spread_run, input_archive, spread_score CASCADE")
    monkeypatch.setenv("DATABASE_URL", url)
    return url


def spread_collection(day, issued="2026-08-15T14:00+00:00", number="G1"):
    ring = [[-125.0, 53.0], [-124.9, 53.0], [-124.9, 53.1], [-125.0, 53.0]]
    feature = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
               "properties": {"band": "highest", "fire_number": number}}
    return {"type": "FeatureCollection", "features": [feature], "date": str(day), "issued": issued}


def test_spread_map_serves_the_latest_run_from_the_database(client, spread_db):
    assert client.get("/api/v1/predict/spread-map").status_code == 503

    today = datetime.now(timezone.utc).date()
    spread_store.save_run(spread_db, spread_collection(today - timedelta(days=2), number="older"))
    spread_store.save_run(spread_db, spread_collection(today - timedelta(days=1)))
    response = client.get("/api/v1/predict/spread-map")
    assert response.headers["content-type"] == "application/geo+json"
    body = response.json()
    assert body["date"] == str(today - timedelta(days=1))
    assert datetime.fromisoformat(body["issued"]) == datetime(2026, 8, 15, 14, tzinfo=timezone.utc)
    assert [f["properties"] for f in body["features"]] == [{"band": "highest", "fire_number": "G1"}]
    assert body["features"][0]["geometry"]["type"] == "Polygon"
    assert body["features"][0]["geometry"]["coordinates"][0][0] == [-125, 53]


def test_rerunning_a_day_replaces_it_and_keeps_empty_days(client, spread_db):
    today = datetime.now(timezone.utc).date()
    spread_store.save_run(spread_db, spread_collection(today))
    spread_store.save_run(spread_db, spread_collection(today) | {"features": []})
    body = client.get("/api/v1/predict/spread-map").json()
    assert body["date"] == str(today)
    assert body["features"] == []


def test_a_failed_save_keeps_the_days_earlier_run(client, spread_db):
    today = datetime.now(timezone.utc).date()
    spread_store.save_run(spread_db, spread_collection(today))
    bad = spread_collection(today)
    bad["features"][0]["properties"]["band"] = "not a band"
    with pytest.raises(psycopg.errors.CheckViolation):
        spread_store.save_run(spread_db, bad)
    assert len(client.get("/api/v1/predict/spread-map").json()["features"]) == 1


def test_spread_map_from_the_database_goes_stale_by_the_runs_date(client, spread_db):
    today = datetime.now(timezone.utc).date()
    spread_store.save_run(spread_db, spread_collection(today - timedelta(days=3)))
    assert client.get("/api/v1/predict/spread-map").status_code == 200
    with psycopg.connect(spread_db) as conn:
        conn.execute("TRUNCATE spread_run CASCADE")
    spread_store.save_run(spread_db, spread_collection(today - timedelta(days=4)))
    assert client.get("/api/v1/predict/spread-map").status_code == 503


def test_a_run_records_the_fires_it_could_not_predict(spread_db):
    spread_store.save_run(spread_db, spread_collection("2026-08-14") | {"failed_fires": ["G2"]})
    with psycopg.connect(spread_db) as conn:
        assert conn.execute("SELECT failed_fires FROM spread_run").fetchone() == (["G2"],)


def test_archive_records_each_file_and_updates_a_rerun(spread_db, tmp_path):
    path = tmp_path / "ecmwf.nc"
    path.write_bytes(b"12345")
    spread_store.save_archive(spread_db, "2026-08-14", {"ecmwf.nc": path})
    path.write_bytes(b"1234567")
    spread_store.save_archive(spread_db, "2026-08-14", {"ecmwf.nc": path})
    with psycopg.connect(spread_db) as conn:
        rows = conn.execute("SELECT source, path, bytes FROM input_archive").fetchall()
    assert rows == [("ecmwf.nc", str(path), 7)]


def spread_score_row(number="G1", new_burn=10, highest=5, elevated=20, hit_highest=4, hit_elevated=3):
    return {"fire_number": number, "unburned_km2": 100, "new_burn_km2": new_burn, "flagged_highest_km2": highest,
            "flagged_elevated_km2": elevated, "hit_highest_km2": hit_highest, "hit_elevated_km2": hit_elevated}


def test_unscored_runs_lists_only_unscored_days_in_range(spread_db):
    for day in ("2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"):
        spread_store.save_run(spread_db, spread_collection(day))
    spread_store.save_scores(spread_db, "2026-08-11", [])
    assert spread_store.unscored_runs(spread_db, "2026-08-11", "2026-08-13") == ["2026-08-12"]


def test_scores_add_up_by_day_and_rerunning_a_prediction_clears_them(spread_db):
    spread_store.save_run(spread_db, spread_collection("2026-08-10"))
    spread_store.save_scores(spread_db, "2026-08-10", [spread_score_row("G1"), spread_score_row("G2")])
    with psycopg.connect(spread_db) as conn:
        row = conn.execute(
            "SELECT fires, new_burn_km2, caught_highest_pct, caught_both_pct, land_highest_pct, land_both_pct "
            "FROM spread_score_daily"
        ).fetchone()
    assert [float(value) for value in row] == [2, 20, 40.0, 70.0, 5.0, 25.0]

    spread_store.save_run(spread_db, spread_collection("2026-08-10"))
    assert spread_store.unscored_runs(spread_db, "2026-08-10", "2026-08-11") == ["2026-08-10"]
    with psycopg.connect(spread_db) as conn:
        assert conn.execute("SELECT count(*) FROM spread_score").fetchone() == (0,)


def test_a_day_with_no_new_burn_scores_without_dividing_by_zero(spread_db):
    spread_store.save_run(spread_db, spread_collection("2026-08-10"))
    spread_store.save_scores(spread_db, "2026-08-10", [spread_score_row(new_burn=0, hit_highest=0, hit_elevated=0)])
    with psycopg.connect(spread_db) as conn:
        assert conn.execute("SELECT caught_highest_pct FROM spread_score_daily").fetchone() == (None,)


def test_run_features_returns_a_days_areas(spread_db):
    spread_store.save_run(spread_db, spread_collection("2026-08-10"))
    features = spread_store.run_features(spread_db, "2026-08-10")
    assert features[0]["properties"] == {"band": "highest", "fire_number": "G1"}
    assert features[0]["geometry"]["type"] == "Polygon"


def test_spread_map_reports_an_unreachable_database(client, monkeypatch):
    def refuse(url):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setenv("DATABASE_URL", "postgresql://nowhere/wildfire")
    monkeypatch.setattr(spread_store.psycopg, "connect", refuse)
    assert client.get("/api/v1/predict/spread-map").status_code == 503


def test_serves_the_web_app_from_the_same_origin(monkeypatch, tmp_path):
    import importlib

    from fastapi.testclient import TestClient

    import api.main

    (tmp_path / "index.html").write_text("<title>BC Wildfire Map</title>")
    monkeypatch.setenv("WEB_DIR", str(tmp_path))
    try:
        client = TestClient(importlib.reload(api.main).app)
        assert "BC Wildfire Map" in client.get("/").text
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/v1").json()["status"] == "ok"
    finally:
        monkeypatch.delenv("WEB_DIR")
        importlib.reload(api.main)


def test_large_responses_are_compressed(client):
    response = client.get("/api/v1/ignition/causes?month=7", headers={"Accept-Encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"


def test_feeds_are_cached_but_failures_are_retried():
    import asyncio

    from api.services.cache import ttl_cache

    calls = []

    @ttl_cache(300)
    async def feed(ok: bool):
        calls.append(ok)
        return {"type": "FeatureCollection", "features": [{"id": 1}]} if ok else None

    async def run():
        for ok in (True, True, False, False):
            await feed(ok)

    asyncio.run(run())
    assert calls == [True, False, False]
