"""CWFIS fire-weather observations at BC stations (WFS GeoJSON, no raster stack)."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

import httpx

from app.cache import TTLCache
from app.config import BC_EAST, BC_NORTH, BC_SOUTH, BC_WEST, get_settings
from app.firms import utc_now

log = logging.getLogger(__name__)

CACHE_KEY = "weather"
weather_cache = TTLCache()
SOURCE_LABEL = "CWFIS public:firewx_stns"


class WeatherUnavailable(Exception):
    pass


def fetch_json(url: str, params: dict, timeout: float) -> dict:
    headers = {"User-Agent": "bc-wildfire-watch/0.1", "Accept": "application/json"}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        text = response.text.lstrip()
        if not text.startswith("{"):
            raise WeatherUnavailable("CWFIS did not return GeoJSON.")
        data = response.json()
        if not isinstance(data, dict):
            raise WeatherUnavailable("CWFIS returned an unexpected payload.")
        return data


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clean_name(value: object) -> str:
    return " ".join(str(value or "").split())


def parse_stations(payload: dict) -> list[dict]:
    """Keep the latest observation per station inside the BC bbox."""
    latest: dict[str, dict] = {}
    for feature in payload.get("features") or []:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        if not isinstance(props, dict):
            continue
        lat = _optional_float(props.get("lat"))
        lon = _optional_float(props.get("lon"))
        fwi = _optional_float(props.get("fwi"))
        if lat is None or lon is None or fwi is None:
            continue
        if not (BC_SOUTH <= lat <= BC_NORTH and BC_WEST <= lon <= BC_EAST):
            continue
        station_id = str(props.get("wmo") or _clean_name(props.get("name")) or "").strip()
        if not station_id:
            continue
        observed_at = str(props.get("rep_date") or "")
        station = {
            "id": station_id,
            "name": _clean_name(props.get("name")) or station_id,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "observed_at": observed_at,
            "temp_c": _optional_float(props.get("temp")),
            "relative_humidity": _optional_float(props.get("rh")),
            "wind_speed_kmh": _optional_float(props.get("ws")),
            "wind_direction_deg": _optional_float(props.get("wdir")),
            "precip_mm": _optional_float(props.get("precip")),
            "ffmc": _optional_float(props.get("ffmc")),
            "dmc": _optional_float(props.get("dmc")),
            "dc": _optional_float(props.get("dc")),
            "bui": _optional_float(props.get("bui")),
            "isi": _optional_float(props.get("isi")),
            "fwi": fwi,
        }
        current = latest.get(station_id)
        if current is None or observed_at >= current["observed_at"]:
            latest[station_id] = station
    stations = list(latest.values())
    stations.sort(key=lambda item: item["name"])
    return stations


def _query_params(since: datetime) -> dict[str, str]:
    stamp = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    cql = f"prov='BC' AND fwi IS NOT NULL AND rep_date AFTER {stamp}"
    return {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "public:firewx_stns",
        "outputFormat": "application/json",
        "count": "4000",
        "sortBy": "rep_date D",
        "CQL_FILTER": cql,
    }


def _empty_failure(detail: str) -> dict:
    return {
        "source": SOURCE_LABEL,
        "available": False,
        "detail": detail,
        "fetched_at": utc_now(),
        "cached": False,
        "station_count": 0,
        "stations": [],
    }


def load_weather() -> dict:
    settings = get_settings()
    cached = weather_cache.get(CACHE_KEY)
    if cached is not None:
        cached["cached"] = True
        return cached

    now = datetime.now(timezone.utc)
    try:
        payload = fetch_json(
            settings.cwfis_wfs_url,
            _query_params(now - timedelta(days=2)),
            settings.request_timeout,
        )
        stations = parse_stations(payload)
        if not stations:
            payload = fetch_json(
                settings.cwfis_wfs_url,
                _query_params(now - timedelta(days=7)),
                settings.request_timeout,
            )
            stations = parse_stations(payload)
    except Exception:
        log.warning("CWFIS fire weather request failed")
        failure = _empty_failure("CWFIS fire weather is unreachable right now.")
        # A configured TTL of 0 means do not cache. `or 60` used to ignore that
        # and pin the failure for a minute.
        failure_ttl = min(60, settings.weather_cache_seconds)
        weather_cache.set(CACHE_KEY, failure, failure_ttl)
        return failure

    result = {
        "source": SOURCE_LABEL,
        "available": True,
        "detail": None if stations else "No recent BC fire-weather observations were returned.",
        "fetched_at": utc_now(),
        "cached": False,
        "station_count": len(stations),
        "stations": stations,
    }
    weather_cache.set(CACHE_KEY, result, settings.weather_cache_seconds)
    return result
