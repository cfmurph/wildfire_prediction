"""Build a FirePayload from a hotspot cluster and ask Grok for a situation report."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.config import get_settings
from app.firms import CACHE_KEY, hotspot_cache, utc_now
from app.geo import compass_label, haversine_km
from app.weather import load_weather
from src.response.grok_client import GrokClient, build_fire_payload

log = logging.getLogger(__name__)

# 375 m VIIRS pixel, converted to hectares, assuming pixels do not overlap.
VIIRS_PIXEL_HA = (0.375 ** 2) * 100
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "live_situation_report.txt"
ACQUIRED_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
DISCLAIMER = (
    "Research prototype for situational awareness. Not an official BC Wildfire Service "
    "product or evacuation order. Next-day spread is not forecast in this view."
)


class MissingGrokKey(Exception):
    pass


class UnknownCluster(Exception):
    pass


class ReportFailed(Exception):
    pass


def footprint_ha(hotspot_count: int) -> float:
    return round(hotspot_count * VIIRS_PIXEL_HA, 1)


def nearest_station(lat: float, lon: float, stations: list[dict]) -> tuple[dict, float] | None:
    best: dict | None = None
    best_distance: float | None = None
    for station in stations:
        distance = haversine_km(lat, lon, station["latitude"], station["longitude"])
        if best_distance is None or distance < best_distance:
            best = station
            best_distance = distance
    if best is None or best_distance is None:
        return None
    return best, best_distance


def _number(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return number


def resolve_cluster(requested: dict) -> dict:
    """Prefer the server-side hotspot cache so clients cannot inflate FRP."""
    cached = hotspot_cache.get(CACHE_KEY)
    if not cached:
        return requested
    for cluster in cached.get("clusters") or []:
        if cluster.get("id") == requested.get("id"):
            return cluster
    raise UnknownCluster(
        "That cluster is not in the current hotspot cache. Reload hotspots and try again."
    )


def _fire_name(cluster: dict, station: dict | None, distance_km: float | None) -> str:
    count = int(cluster["hotspot_count"])
    parts = [f"{count} VIIRS detections"]
    if cluster.get("max_frp") is not None:
        parts.append(f"max FRP {_number(cluster['max_frp']):.1f} MW")
    if cluster.get("mean_frp") is not None:
        parts.append(f"mean FRP {_number(cluster['mean_frp']):.1f} MW")
    satellites = [str(item) for item in cluster.get("satellites") or [] if item]
    if satellites:
        parts.append("satellites " + ", ".join(satellites[:6]))
    confidence = cluster.get("confidence") or {}
    parts.append(
        "confidence high {high}, nominal {nominal}, low {low}".format(
            high=int(confidence.get("high") or 0),
            nominal=int(confidence.get("nominal") or 0),
            low=int(confidence.get("low") or 0),
        )
    )
    if station is not None and distance_km is not None:
        parts.append(f"nearest CWFIS station {station['name']} ({distance_km:.0f} km)")
    else:
        parts.append("no CWFIS fire-weather station available")
    parts.append("approximate footprint assumes non-overlapping 375 m pixels")
    parts.append("ML spread model is not connected")
    return "; ".join(parts)


def build_live_payload(cluster: dict) -> tuple[object, dict | None, float | None]:
    weather = load_weather()
    station: dict | None = None
    distance: float | None = None
    if weather.get("available") and weather.get("stations"):
        found = nearest_station(cluster["latitude"], cluster["longitude"], weather["stations"])
        if found is not None:
            station, distance = found

    acquired = cluster.get("latest_acquisition") or ""
    as_of = acquired if isinstance(acquired, str) and ACQUIRED_RE.match(acquired) else utc_now()
    wind_deg = _number(station.get("wind_direction_deg") if station else None)
    wind_speed = _number(station.get("wind_speed_kmh") if station else None)
    safe_id = re.sub(r"[^0-9.\-]", "_", str(cluster.get("id") or "cluster"))[:40]

    payload = build_fire_payload(
        fire_id=f"VIIRS-{safe_id}",
        fire_name=_fire_name(cluster, station, distance),
        province="BC",
        as_of_datetime=as_of,
        current_area_ha=footprint_ha(int(cluster["hotspot_count"])),
        ml_spread_probs={"p25": 0.0, "p50": 0.0, "p75": 0.0},
        fwi_values={
            "FWI": _number(station.get("fwi") if station else None),
            "ISI": _number(station.get("isi") if station else None),
            "BUI": _number(station.get("bui") if station else None),
            "FFMC": _number(station.get("ffmc") if station else None),
        },
        wind_values={
            "speed_kmh": wind_speed,
            "direction_deg": wind_deg,
            "direction_label": compass_label(wind_deg),
        },
        risk_zones=[],
        burnp3_max=None,
    )
    return payload, station, distance


def public_station(station: dict, distance_km: float) -> dict:
    return {
        "id": station["id"],
        "name": station["name"],
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "observed_at": station.get("observed_at"),
        "distance_km": round(distance_km, 1),
        "fwi": station.get("fwi"),
        "isi": station.get("isi"),
        "bui": station.get("bui"),
        "ffmc": station.get("ffmc"),
        "wind_speed_kmh": station.get("wind_speed_kmh"),
        "wind_direction_deg": station.get("wind_direction_deg"),
        "temp_c": station.get("temp_c"),
        "relative_humidity": station.get("relative_humidity"),
    }


def generate_report(requested: dict) -> dict:
    settings = get_settings()
    if not settings.xai_api_key:
        raise MissingGrokKey(
            "XAI_API_KEY is not configured. Add it to the API environment to generate "
            "situation reports."
        )

    cluster = resolve_cluster(requested)
    payload, station, distance = build_live_payload(cluster)
    try:
        client = GrokClient(
            api_key=settings.xai_api_key,
            model=settings.grok_model,
            max_tokens=700,
            temperature=0.2,
            situation_report_prompt=str(PROMPT_PATH),
            timeout=30,
            max_retries=2,
        )
        report = client.situation_report(payload)
    except Exception as exc:
        log.warning("Situation report failed (%s)", type(exc).__name__)
        raise ReportFailed(
            "Situation report failed. Verify XAI_API_KEY and try again."
        ) from exc

    weather_public = None
    if station is not None and distance is not None:
        weather_public = public_station(station, distance)

    return {
        "report": report,
        "model": settings.grok_model,
        "ml_forecast_available": False,
        "approximate_area_ha": payload.current_area_ha,
        "fire_weather": weather_public,
        "disclaimer": DISCLAIMER,
        "inputs": {
            "fire_id": payload.fire_id,
            "as_of": payload.as_of_datetime,
            "approximate_area_ha": payload.current_area_ha,
            "fwi": payload.fwi_summary.FWI,
            "isi": payload.fwi_summary.ISI,
            "bui": payload.fwi_summary.BUI,
            "ffmc": payload.fwi_summary.FFMC,
            "wind_speed_kmh": payload.wind_summary.speed_kmh,
            "wind_direction": compass_label(payload.wind_summary.direction_deg),
            "station_name": station["name"] if station else None,
        },
    }
