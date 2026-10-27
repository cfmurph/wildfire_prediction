"""POST /report validation, Grok failures, and the zero-spread contract."""

from __future__ import annotations

import json

import openai
import pytest
from app.reporting import PROMPT_PATH, footprint_ha
from app.schemas import ClusterIn

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


def test_prompt_forbids_invented_forecasts_and_has_one_placeholder():
    text = PROMPT_PATH.read_text()
    assert text.count("{") == 1
    assert "{payload_json}" in text
    lowered = text.lower()
    assert "placeholders at zero" in lowered
    assert "do not describe them as a forecast" in lowered
    assert "do not invent" in lowered
    assert "not connected" in lowered
    assert "not an official evacuation order" in lowered
    rendered = text.format(payload_json='{"spread_probabilities": {"p50_area_ha": 0.0}}')
    assert "p50_area_ha" in rendered


def test_health_shape_never_includes_secrets(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "super-secret-firms")
    monkeypatch.setenv("XAI_API_KEY", "super-secret-xai")
    response = client.get("/health")
    body = response.json()
    assert response.status_code == 200
    assert set(body) == {
        "status",
        "region",
        "firms_configured",
        "grok_configured",
        "views",
        "planned",
    }
    assert body["firms_configured"] is True
    assert body["grok_configured"] is True
    assert body["planned"]["predict"].startswith("Next-day spread")
    blob = response.text
    assert "super-secret-firms" not in blob
    assert "super-secret-xai" not in blob


def test_planned_routes_reject_writes_and_keep_the_501_contract(client):
    for path, view in (("/history", "history"), ("/risk", "risk"), ("/predict", "predict")):
        response = client.get(path)
        assert response.status_code == 501
        assert set(response.json()) == {"view", "status", "detail"}
        assert response.json()["view"] == view
        assert response.json()["status"] == "coming_soon"
        denied = client.post(path, json={})
        assert denied.status_code == 405
    assert client.get("/report").status_code == 405
    assert client.post("/hotspots").status_code == 405


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"cluster": None},
        {"cluster": {}},
        {"cluster": {"id": "", "latitude": 49.1, "longitude": -123.1, "hotspot_count": 1}},
        {"cluster": {"id": "x" * 81, "latitude": 49.1, "longitude": -123.1, "hotspot_count": 1}},
        {"cluster": {"id": "x", "latitude": 90, "longitude": -123.1, "hotspot_count": 1}},
        {"cluster": {"id": "x", "latitude": 49.1, "longitude": 0, "hotspot_count": 1}},
        {"cluster": {"id": "x", "latitude": 49.1, "longitude": -123.1, "hotspot_count": 0}},
        {"cluster": {"id": "x", "latitude": 49.1, "longitude": -123.1, "hotspot_count": 100_001}},
        {"cluster": {"id": "x", "latitude": 49.1, "longitude": -123.1, "hotspot_count": 1, "max_frp": -1}},
        {"cluster": {"id": "x", "latitude": "nan", "longitude": -123.1, "hotspot_count": 1}},
        {"cluster": {"id": "x", "latitude": 49.1, "longitude": "-inf", "hotspot_count": 1}},
    ],
)
def test_report_rejects_invalid_clusters(client, payload):
    response = client.post("/report", json=payload)
    assert response.status_code == 422
    assert "super-secret" not in response.text


def test_report_ignores_extra_and_sanitizes_satellites(client):
    noisy = {
        "cluster": {
            **CLUSTER,
            "satellites": ["N", "<script>alert(1)</script>", "NOAA-21-EXTRA-LONG", "ok"]
            + [f"s{i}" for i in range(10)],
            "confidence": {"high": "nope", "nominal": 2, "evil": 99},
            "note": "x" * 100_000,
        },
        "unused": "y" * 50_000,
    }
    response = client.post("/report", json=noisy)
    assert response.status_code == 503
    assert "XAI_API_KEY" in response.json()["detail"]
    assert "<script>" not in response.text
    parsed = ClusterIn.model_validate(noisy["cluster"])
    assert all("<" not in item and len(item) <= 12 for item in parsed.satellites)
    assert len(parsed.satellites) <= 8
    assert "evil" not in parsed.confidence
    assert parsed.confidence["high"] == 0
    assert parsed.confidence["nominal"] == 2
    assert ClusterIn.model_validate({**CLUSTER, "satellites": "N"}).satellites == []


class _OpenAI:
    captured: dict = {}

    def __init__(self, **kwargs):
        _OpenAI.captured["init"] = kwargs
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        _OpenAI.captured["create"] = kwargs
        message = type("Message", (), {"content": "Grounded briefing."})()
        choice = type("Choice", (), {"message": message})()
        return type("Resp", (), {"choices": [choice]})()


class _FailingOpenAI(_OpenAI):
    def create(self, **_kwargs):
        raise TimeoutError("timed out with super-secret-xai in the request")


def test_report_prompt_keeps_spread_at_zero(client, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "super-secret-xai")
    monkeypatch.setenv("GROK_MODEL", "grok-3-mini")
    monkeypatch.setattr(openai, "OpenAI", _OpenAI)
    monkeypatch.setattr(
        "app.reporting.load_weather",
        lambda: {"available": False, "stations": []},
    )
    _OpenAI.captured = {}
    body = dict(CLUSTER)
    body["latest_acquisition"] = "not-a-timestamp"
    response = client.post("/report", json={"cluster": body})
    assert response.status_code == 200
    payload_text = response.text
    assert "super-secret-xai" not in payload_text
    assert response.json()["ml_forecast_available"] is False
    prompt = _OpenAI.captured["create"]["messages"][0]["content"]
    assert "Do not describe them as a forecast" in prompt
    start = prompt.index("{")
    end = prompt.rindex("}") + 1
    sent = json.loads(prompt[start:end])
    assert sent["spread_probabilities"] == {
        "p25_area_ha": 0.0,
        "p50_area_ha": 0.0,
        "p75_area_ha": 0.0,
    }
    assert sent["high_risk_zones"] == []
    assert sent["current_area_ha"] == footprint_ha(4)
    assert _OpenAI.captured["init"]["api_key"] == "super-secret-xai"
    assert "not-a-timestamp" not in sent["as_of_datetime"]


def test_report_grok_timeout_hides_secrets(client, monkeypatch, caplog):
    monkeypatch.setenv("XAI_API_KEY", "super-secret-xai")
    monkeypatch.setattr(openai, "OpenAI", _FailingOpenAI)
    monkeypatch.setattr("src.response.grok_client.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "app.reporting.load_weather",
        lambda: {"available": False, "stations": []},
    )
    response = client.post("/report", json={"cluster": CLUSTER})
    assert response.status_code == 502
    assert "super-secret-xai" not in response.text
    assert "timed out with" not in response.text
    assert "Situation report failed" in response.json()["detail"]
    assert "super-secret-xai" not in caplog.text


def test_report_uses_nearest_station_not_the_first(client, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "super-secret-xai")
    monkeypatch.setattr(openai, "OpenAI", _OpenAI)
    _OpenAI.captured = {}
    monkeypatch.setattr(
        "app.reporting.load_weather",
        lambda: {
            "available": True,
            "stations": [
                {
                    "id": "far",
                    "name": "FAR STATION",
                    "latitude": 60.0,
                    "longitude": -139.0,
                    "observed_at": "2026-09-26T12:00:00Z",
                    "fwi": 30,
                    "isi": 1,
                    "bui": 1,
                    "ffmc": 1,
                    "wind_speed_kmh": 1,
                    "wind_direction_deg": 90,
                },
                {
                    "id": "near",
                    "name": "NEAR STATION",
                    "latitude": 49.11,
                    "longitude": -123.11,
                    "observed_at": "2026-09-26T12:00:00Z",
                    "fwi": 11,
                    "isi": 4,
                    "bui": 5,
                    "ffmc": 80,
                    "wind_speed_kmh": 18,
                    "wind_direction_deg": 22.5,
                },
            ],
        },
    )
    response = client.post("/report", json={"cluster": CLUSTER})
    assert response.status_code == 200
    assert response.json()["fire_weather"]["name"] == "NEAR STATION"
    assert response.json()["inputs"]["wind_direction"] == "NE"
    assert response.json()["inputs"]["fwi"] == 11
    prompt = _OpenAI.captured["create"]["messages"][0]["content"]
    assert "NEAR STATION" in prompt
    assert "FAR STATION" not in prompt
