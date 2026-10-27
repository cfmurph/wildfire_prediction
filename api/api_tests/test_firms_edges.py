"""FIRMS parsing, upstream failures, source selection, and acquisition timestamps."""

from __future__ import annotations

import httpx
import pytest
from api_tests.conftest import REAL_FETCH_TEXT, REAL_HTTPX_CLIENT
from app.firms import (
    InvalidMapKey,
    UnexpectedFirmsPayload,
    _acquired_at,
    cluster_hotspots,
    parse_firms_csv,
)

HEADER = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight"
)


def _csv(*rows: str) -> str:
    return HEADER + "\n" + "\n".join(rows) + ("\n" if rows else "")


def _line(
    lat: str = "49.10",
    lon: str = "-123.10",
    bright: str = "340.0",
    date: str = "2026-09-26",
    time: str = "1900",
    satellite: str = "N",
    confidence: str = "high",
    frp: str = "10.0",
    daynight: str = "D",
) -> str:
    return (
        f"{lat},{lon},{bright},0.4,0.4,{date},{time},{satellite},"
        f"VIIRS,{confidence},2.0NRT,290.0,{frp},{daynight}"
    )


def test_empty_whitespace_and_bom_csv_are_empty():
    assert parse_firms_csv("") == []
    assert parse_firms_csv("   \n\n") == []
    assert parse_firms_csv("\ufeff") == []
    assert parse_firms_csv("\ufeff" + HEADER + "\n") == []


def test_missing_columns_keep_the_point_and_drop_optional_fields():
    rows = parse_firms_csv("latitude,longitude\n49.10,-123.10\n")
    assert len(rows) == 1
    assert rows[0]["frp"] is None
    assert rows[0]["brightness"] is None
    assert rows[0]["confidence"] == "unknown"
    assert rows[0]["satellite"] == ""
    assert rows[0]["acquired_at"] is None


def test_malformed_rows_are_skipped_and_later_rows_kept():
    text = _csv(
        "not-a-lat,-123.10,1,0.4,0.4,2026-09-26,1900,N,VIIRS,high,2,290,5,D",
        _line(lat="49.20", frp="nan"),
        _line(lat="49.40", lon="-123.40", frp="inf"),
        _line(lat="49.50", lon="-123.50", frp=""),
        _line(lat="49.60", lon="-123.60", frp="7.5"),
    )
    rows = parse_firms_csv(text)
    assert [row["latitude"] for row in rows] == [49.2, 49.4, 49.5, 49.6]
    assert rows[0]["frp"] is None
    assert rows[1]["frp"] is None
    assert rows[2]["frp"] is None
    assert rows[3]["frp"] == 7.5


def test_points_outside_the_bc_bbox_are_dropped_and_edges_are_kept():
    text = _csv(
        _line(lat="48.30", lon="-139.06"),
        _line(lat="60.00", lon="-114.03"),
        _line(lat="48.29", lon="-125.00"),
        _line(lat="60.01", lon="-125.00"),
        _line(lat="54.00", lon="-139.07"),
        _line(lat="54.00", lon="-114.02"),
        _line(lat="0", lon="0"),
    )
    rows = parse_firms_csv(text)
    assert {(row["latitude"], row["longitude"]) for row in rows} == {
        (48.30, -139.06),
        (60.00, -114.03),
    }


def test_duplicate_detections_and_confidence_daynight_aliases():
    text = _csv(
        _line(confidence="H", daynight="d"),
        _line(confidence="H", daynight="d"),
        _line(lat="49.30", confidence="n", daynight="N"),
        _line(lat="49.40", confidence="low", daynight="x"),
        _line(lat="49.50", confidence="85", daynight=""),
    )
    rows = parse_firms_csv(text)
    assert len(rows) == 4
    assert rows[0]["confidence"] == "high"
    assert rows[0]["daynight"] == "D"
    assert rows[1]["confidence"] == "nominal"
    assert rows[2]["confidence"] == "low"
    assert rows[2]["daynight"] == ""
    assert rows[3]["confidence"] == "unknown"


def test_bad_clocks_and_dates_do_not_become_timestamps():
    assert _acquired_at("2026-09-26", "930") == "2026-09-26T09:30:00Z"
    assert _acquired_at("2026-09-26", "93") is None
    assert _acquired_at("2026-09-26", "2460") is None
    assert _acquired_at("2026-13-40", "1200") is None
    assert _acquired_at("2026/09/26", "1200") is None
    assert _acquired_at("26", "1200") is None
    text = _csv(_line(time="2460"), _line(lat="50.00", date="2026-02-30", time="1200"))
    rows = parse_firms_csv(text)
    assert rows[0]["acquired_at"] is None
    assert rows[1]["acquired_at"] is None


def test_non_csv_payloads_and_rejected_keys():
    with pytest.raises(InvalidMapKey):
        parse_firms_csv("Invalid MAP_KEY.")
    with pytest.raises(InvalidMapKey):
        parse_firms_csv("<html>invalid key</html>")
    with pytest.raises(UnexpectedFirmsPayload):
        parse_firms_csv("<html>upstream timeout</html>")
    with pytest.raises(UnexpectedFirmsPayload):
        parse_firms_csv("invalid format, please retry")


def test_clustering_cells_frp_order_and_missing_frp():
    rows = parse_firms_csv(
        _csv(
            _line(lat="49.00", lon="-123.19", frp="5", satellite="N21"),
            _line(lat="49.19", lon="-123.05", frp="15", satellite="N"),
            _line(lat="49.20", lon="-123.00", frp=""),
            _line(lat="53.00", lon="-132.19", frp="1", time="0100"),
            _line(lat="53.05", lon="-132.10", frp="3", time="0200", satellite="N20"),
        )
    )
    clusters = cluster_hotspots(rows)
    by_id = {item["id"]: item for item in clusters}
    coast = by_id["49.00,-123.20"]
    assert coast["hotspot_count"] == 2
    assert coast["max_frp"] == 15.0
    assert coast["mean_frp"] == 10.0
    assert coast["satellites"] == ["N", "N21"]
    assert by_id["49.20,-123.00"]["max_frp"] is None
    north = by_id["53.00,-132.20"]
    assert north["hotspot_count"] == 2
    assert north["latest_acquisition"] == "2026-09-26T02:00:00Z"
    assert clusters[0]["max_frp"] == 15.0
    assert clusters[-1]["max_frp"] is None


def _client(handler):
    transport = httpx.MockTransport(handler)

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return REAL_HTTPX_CLIENT(*args, **kwargs)

    return factory


def test_fetch_text_maps_rejected_keys_and_raises_on_http_errors():
    def rejected(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Invalid MAP_KEY")

    client = _client(rejected)
    # Call the real function; the autouse fixture replaces the module global.
    import app.firms as firms

    firms.httpx.Client = client
    try:
        with pytest.raises(InvalidMapKey):
            REAL_FETCH_TEXT("https://firms.example/secret-key/data", 5.0)
    finally:
        firms.httpx.Client = REAL_HTTPX_CLIENT

    def down(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    firms.httpx.Client = _client(down)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            REAL_FETCH_TEXT("https://firms.example/secret-key/data", 5.0)
    finally:
        firms.httpx.Client = REAL_HTTPX_CLIENT

    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    firms.httpx.Client = _client(timeout)
    try:
        with pytest.raises(httpx.TimeoutException):
            REAL_FETCH_TEXT("https://firms.example/secret-key/data", 5.0)
    finally:
        firms.httpx.Client = REAL_HTTPX_CLIENT


def test_hotspots_route_covers_sources_day_range_and_upstream(client, monkeypatch, caplog):
    monkeypatch.setenv("FIRMS_MAP_KEY", "super-secret-firms")
    monkeypatch.setenv("FIRMS_DAY_RANGE", "9")
    monkeypatch.setenv("FIRMS_SOURCES", "MODIS_NRT, VIIRS_SNPP_NRT ,VIIRS_NOAA21_NRT")
    calls: list[str] = []

    def fake(url: str, timeout: float) -> str:
        calls.append(url)
        assert timeout == 30
        if "VIIRS_NOAA21_NRT" in url:
            raise httpx.ReadTimeout("timed out")
        if "MODIS" in url:
            raise AssertionError("disallowed source was requested")
        return _csv(_line())

    monkeypatch.setattr("app.firms.fetch_text", fake)
    response = client.get("/hotspots")
    assert response.status_code == 200
    body = response.json()
    assert body["day_range"] == 5
    assert body["sources"] == ["VIIRS_SNPP_NRT", "VIIRS_NOAA21_NRT"]
    assert body["hotspot_count"] == 1
    assert body["warnings"] == ["VIIRS_NOAA21_NRT is temporarily unavailable."]
    assert len(calls) == 2
    assert all(url.endswith("/5") for url in calls)
    assert "super-secret-firms" not in response.text
    assert "super-secret-firms" not in caplog.text


def test_invalid_day_range_and_unknown_sources_fall_back(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "k")
    monkeypatch.setenv("FIRMS_DAY_RANGE", "nope")
    monkeypatch.setenv("FIRMS_SOURCES", "MODIS_NRT,NOT_A_SOURCE")
    calls: list[str] = []

    def fake(url: str, _timeout: float) -> str:
        calls.append(url)
        return _csv()

    monkeypatch.setattr("app.firms.fetch_text", fake)
    response = client.get("/hotspots")
    assert response.status_code == 200
    assert response.json()["day_range"] == 2
    assert response.json()["hotspot_count"] == 0
    assert response.json()["sources"] == [
        "VIIRS_SNPP_NRT",
        "VIIRS_NOAA20_NRT",
        "VIIRS_NOAA21_NRT",
    ]
    assert len(calls) == 3
    assert all(url.rsplit("/", 1)[-1] == "2" for url in calls)


def test_day_range_lower_bound_and_custom_base_url(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "map-key-zzz")
    monkeypatch.setenv("FIRMS_DAY_RANGE", "0")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT")
    monkeypatch.setenv("FIRMS_BASE_URL", "https://firms.example/root/")
    seen: list[str] = []

    def fake(url: str, _timeout: float) -> str:
        seen.append(url)
        return "Invalid key"

    monkeypatch.setattr("app.firms.fetch_text", fake)
    rejected = client.get("/hotspots")
    assert rejected.status_code == 503
    assert seen[0].startswith(
        "https://firms.example/root/api/area/csv/map-key-zzz/VIIRS_SNPP_NRT/"
    )
    assert seen[0].endswith("/1")
    assert "map-key-zzz" not in rejected.text


def test_rejected_key_stops_remaining_sources(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "secret-map")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT")
    calls: list[str] = []

    def fake(url: str, _timeout: float) -> str:
        calls.append(url)
        return "Invalid MAP_KEY"

    monkeypatch.setattr("app.firms.fetch_text", fake)
    response = client.get("/hotspots")
    assert response.status_code == 503
    assert len(calls) == 1
    assert "secret-map" not in response.text


def test_all_sources_http_500_is_bad_gateway_without_the_key(client, monkeypatch, caplog):
    monkeypatch.setenv("FIRMS_MAP_KEY", "secret-map")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT")

    def fake(url: str, _timeout: float) -> str:
        request = httpx.Request("GET", url)
        response = httpx.Response(500, request=request, text="boom")
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr("app.firms.fetch_text", fake)
    response = client.get("/hotspots")
    assert response.status_code == 502
    assert "secret-map" not in response.text
    assert "secret-map" not in caplog.text


def test_missing_key_does_not_call_upstream(client, monkeypatch):
    calls = {"n": 0}

    def fake(_url: str, _timeout: float) -> str:
        calls["n"] += 1
        return _csv()

    monkeypatch.setattr("app.firms.fetch_text", fake)
    response = client.get("/hotspots")
    assert response.status_code == 503
    assert calls["n"] == 0


def test_negative_and_blank_day_range(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "k")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT")
    monkeypatch.setenv("FIRMS_DAY_RANGE", " -4 ")
    seen: list[str] = []

    def fake(url: str, _timeout: float) -> str:
        seen.append(url)
        return _csv()

    monkeypatch.setattr("app.firms.fetch_text", fake)
    assert client.get("/hotspots").json()["day_range"] == 1
    assert seen[0].endswith("/1")
