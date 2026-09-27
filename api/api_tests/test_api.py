"""API tests with mocked FIRMS, CWFIS, and Grok."""

from __future__ import annotations

from app.firms import hotspot_cache, parse_firms_csv
from app.geo import haversine_km
from app.reporting import PROMPT_PATH, footprint_ha
from app.weather import parse_stations

FIRMS_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "53.25,-132.12,320.5,0.4,0.4,2026-09-26,1842,N,VIIRS,high,2.0NRT,290.1,15.2,N\n"
    "53.26,-132.11,310.0,0.4,0.4,2026-09-26,1842,N,VIIRS,nominal,2.0NRT,285.0,8.1,N\n"
    "49.10,-123.10,340.0,0.5,0.4,2026-09-26,1900,N21,VIIRS,high,2.0NRT,300.0,40.0,D\n"
    "49.10,-123.10,340.0,0.5,0.4,2026-09-26,1900,N21,VIIRS,high,2.0NRT,300.0,40.0,D\n"
    "0,0,300,0.4,0.4,2026-09-26,1900,N,VIIRS,low,2.0NRT,290,5,D\n"
    "not,a,number,0,0,2026-09-26,1900,N,VIIRS,low,2.0NRT,1,1,D\n"
)

CLUSTER = {
    "id": "49.00,-123.20",
    "latitude": 49.1,
    "longitude": -123.1,
    "hotspot_count": 4,
    "max_frp": 12.5,
    "mean_frp": 6.0,
    "max_brightness": 330.0,
    "latest_acquisition": "2026-09-26T19:00:00Z",
    "confidence": {"high": 3, "nominal": 1, "low": 0, "unknown": 0},
    "satellites": ["N", "N21"],
    "daynight": {"D": 3, "N": 1},
}


def _station(**overrides):
    base = {
        "wmo": 100001,
        "name": "  TEST   STATION  ",
        "rep_date": "2026-09-26T12:00:00Z",
        "prov": "BC",
        "lat": 49.1,
        "lon": -123.1,
        "fwi": 18.5,
        "ffmc": 90.0,
        "dmc": 40.0,
        "dc": 300.0,
        "bui": 55.0,
        "isi": 8.0,
        "ws": 22.0,
        "wdir": 315.0,
        "temp": 20.0,
        "rh": 30.0,
        "precip": 0.0,
    }
    base.update(overrides)
    return {"type": "Feature", "properties": base}


class FakeGrok:
    captured: dict = {}

    def __init__(self, **kwargs):
        FakeGrok.captured["kwargs"] = kwargs

    def situation_report(self, payload):
        FakeGrok.captured["payload"] = payload
        return "A calm briefing about the hotspot cluster."


def test_health_reports_key_presence_without_leaking_secrets(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "super-secret-firms")
    response = client.get("/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["region"] == "BC"
    assert body["firms_configured"] is True
    assert body["grok_configured"] is False
    assert "super-secret-firms" not in response.text


def test_hotspots_missing_key(client):
    response = client.get("/hotspots")
    assert response.status_code == 503
    assert "FIRMS_MAP_KEY" in response.json()["detail"]


def test_hotspots_parse_cluster_and_cache(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "test-map-key")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT")
    calls = {"n": 0}

    def fake(url: str, timeout: float) -> str:
        calls["n"] += 1
        assert "test-map-key" in url
        assert "VIIRS_SNPP_NRT" in url
        assert "-139.06,48.30,-114.03,60.00" in url
        assert timeout > 0
        return FIRMS_CSV

    monkeypatch.setattr("app.firms.fetch_text", fake)
    first = client.get("/hotspots")
    second = client.get("/hotspots")
    assert first.status_code == 200
    body = first.json()
    assert body["cached"] is False
    assert body["hotspot_count"] == 3
    assert body["cluster_count"] == 2
    assert body["clusters"][0]["id"] == "49.00,-123.20"
    assert body["clusters"][0]["max_frp"] == 40.0
    assert body["clusters"][0]["hotspot_count"] == 1
    haida = body["clusters"][1]
    assert haida["id"] == "53.20,-132.20"
    assert haida["hotspot_count"] == 2
    assert haida["max_frp"] == 15.2
    assert haida["mean_frp"] == 11.65
    assert haida["latitude"] == 53.255
    assert haida["confidence"]["high"] == 1
    assert haida["confidence"]["nominal"] == 1
    assert haida["latest_acquisition"] == "2026-09-26T18:42:00Z"
    assert second.json()["cached"] is True
    assert calls["n"] == 1
    assert "test-map-key" not in first.text


def test_hotspots_invalid_key_and_upstream_failure(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "test-map-key")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT")

    def invalid(_url: str, _timeout: float) -> str:
        return "Invalid MAP_KEY"

    monkeypatch.setattr("app.firms.fetch_text", invalid)
    rejected = client.get("/hotspots")
    assert rejected.status_code == 503
    assert "test-map-key" not in rejected.text

    hotspot_cache.clear()

    def down(url: str, _timeout: float) -> str:
        raise RuntimeError(f"boom {url}")

    monkeypatch.setattr("app.firms.fetch_text", down)
    failed = client.get("/hotspots")
    assert failed.status_code == 502
    assert "test-map-key" not in failed.text


def test_hotspots_dedupes_identical_rows_across_sources(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "test-map-key")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT")

    def fake(_url: str, _timeout: float) -> str:
        return FIRMS_CSV

    monkeypatch.setattr("app.firms.fetch_text", fake)
    response = client.get("/hotspots")
    assert response.status_code == 200
    assert response.json()["hotspot_count"] == 3
    assert response.json()["cluster_count"] == 2


def test_hotspots_partial_source_warning(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "test-map-key")
    monkeypatch.setenv("FIRMS_SOURCES", "VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT")

    def fake(url: str, _timeout: float) -> str:
        if "VIIRS_NOAA20_NRT" in url:
            raise RuntimeError("sensor down")
        return FIRMS_CSV

    monkeypatch.setattr("app.firms.fetch_text", fake)
    response = client.get("/hotspots")
    assert response.status_code == 200
    assert response.json()["warnings"] == ["VIIRS_NOAA20_NRT is temporarily unavailable."]
    assert response.json()["cluster_count"] == 2


def test_parse_empty_and_header_only():
    assert parse_firms_csv("") == []
    assert parse_firms_csv("latitude,longitude\n") == []


def test_haversine_one_tenth_degree():
    distance = haversine_km(53.0, -132.0, 53.1, -132.0)
    assert 10 < distance < 12


def test_weather_dedupes_and_skips_bad_rows():
    payload = {
        "features": [
            _station(rep_date="2026-09-25T12:00:00Z", fwi=4),
            _station(rep_date="2026-09-26T12:00:00Z", fwi=18.5),
            _station(wmo=100002, name="OUTSIDE", lat=10, lon=10, fwi=30),
            _station(wmo=100003, name="NO FWI", fwi=None),
        ]
    }
    stations = parse_stations(payload)
    assert len(stations) == 1
    assert stations[0]["name"] == "TEST STATION"
    assert stations[0]["fwi"] == 18.5
    assert stations[0]["wind_direction_deg"] == 315.0


def test_weather_route_cache_and_widen(client, monkeypatch):
    calls = {"n": 0}

    def fake(_url, params, _timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            assert "AFTER" in params["CQL_FILTER"]
            return {"features": []}
        return {"features": [_station()]}

    monkeypatch.setattr("app.weather.fetch_json", fake)
    first = client.get("/weather")
    assert first.status_code == 200
    assert first.json()["available"] is True
    assert first.json()["station_count"] == 1
    assert first.json()["cached"] is False
    assert calls["n"] == 2
    second = client.get("/weather")
    assert second.json()["cached"] is True
    assert calls["n"] == 2


def test_weather_unreachable_is_soft_failure(client, monkeypatch):
    def fake(_url, _params, _timeout):
        raise RuntimeError("cwfis down")

    monkeypatch.setattr("app.weather.fetch_json", fake)
    response = client.get("/weather")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["stations"] == []
    assert "CWFIS" in body["detail"]


def test_report_missing_key(client, monkeypatch):
    FakeGrok.captured = {}
    monkeypatch.setattr("app.reporting.GrokClient", FakeGrok)
    response = client.post("/report", json={"cluster": CLUSTER})
    assert response.status_code == 503
    assert "XAI_API_KEY" in response.json()["detail"]
    assert "payload" not in FakeGrok.captured


def test_report_uses_weather_and_grok(client, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai")
    monkeypatch.setenv("GROK_MODEL", "grok-3-mini")
    FakeGrok.captured = {}
    monkeypatch.setattr("app.reporting.GrokClient", FakeGrok)
    monkeypatch.setattr(
        "app.reporting.load_weather",
        lambda: {
            "available": True,
            "stations": [
                {
                    "id": "100001",
                    "name": "TEST STATION",
                    "latitude": 49.1,
                    "longitude": -123.1,
                    "observed_at": "2026-09-26T12:00:00Z",
                    "temp_c": 20.0,
                    "relative_humidity": 30.0,
                    "wind_speed_kmh": 22.0,
                    "wind_direction_deg": 315.0,
                    "precip_mm": 0.0,
                    "ffmc": 90.0,
                    "dmc": 40.0,
                    "dc": 300.0,
                    "bui": 55.0,
                    "isi": 8.0,
                    "fwi": 18.5,
                }
            ],
        },
    )
    response = client.post("/report", json={"cluster": CLUSTER})
    assert response.status_code == 200
    body = response.json()
    assert body["report"].startswith("A calm briefing")
    assert body["ml_forecast_available"] is False
    assert body["model"] == "grok-3-mini"
    assert body["fire_weather"]["name"] == "TEST STATION"
    assert body["inputs"]["fwi"] == 18.5
    assert body["inputs"]["wind_direction"] == "NW"
    assert body["inputs"]["wind_speed_kmh"] == 22.0
    payload = FakeGrok.captured["payload"]
    assert payload.province == "BC"
    assert payload.current_area_ha == footprint_ha(4)
    assert payload.spread_probabilities.p50_area_ha == 0
    assert payload.high_risk_zones == []
    assert payload.fwi_summary.FWI == 18.5
    assert "ML spread model is not connected" in payload.fire_name
    assert "max FRP 12.5 MW" in payload.fire_name
    assert FakeGrok.captured["kwargs"]["situation_report_prompt"].endswith(
        "prompts/live_situation_report.txt"
    )
    assert "test-xai" not in response.text


def test_report_prefers_cached_cluster_over_client_numbers(client, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai")
    FakeGrok.captured = {}
    monkeypatch.setattr("app.reporting.GrokClient", FakeGrok)
    monkeypatch.setattr(
        "app.reporting.load_weather",
        lambda: {"available": False, "stations": []},
    )
    cached = dict(CLUSTER)
    cached["max_frp"] = 10.0
    cached["hotspot_count"] = 4
    hotspot_cache.set("hotspots", {"clusters": [cached]}, 600)
    forged = dict(CLUSTER)
    forged["max_frp"] = 999
    forged["hotspot_count"] = 4
    response = client.post("/report", json={"cluster": forged})
    assert response.status_code == 200
    assert "max FRP 10.0 MW" in FakeGrok.captured["payload"].fire_name
    assert "999" not in FakeGrok.captured["payload"].fire_name
    assert "no CWFIS fire-weather station available" in FakeGrok.captured["payload"].fire_name


def test_report_unknown_cluster_when_cache_is_warm(client, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai")
    hotspot_cache.set("hotspots", {"clusters": [dict(CLUSTER)]}, 600)
    other = dict(CLUSTER)
    other["id"] = "50.00,-121.00"
    response = client.post("/report", json={"cluster": other})
    assert response.status_code == 404


def test_live_prompt_has_a_single_placeholder():
    text = PROMPT_PATH.read_text()
    assert text.count("{") == 1
    assert "{payload_json}" in text
    rendered = text.format(payload_json='{"ok": true}')
    assert '{"ok": true}' in rendered
