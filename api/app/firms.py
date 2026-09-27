"""NASA FIRMS VIIRS near-real-time hotspots for the British Columbia bbox."""

from __future__ import annotations

import csv
import io
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

import httpx

from app.cache import TTLCache
from app.config import BC_EAST, BC_NORTH, BC_SOUTH, BC_WEST, get_settings

log = logging.getLogger(__name__)

CLUSTER_CELL_DEG = 0.2
CACHE_KEY = "hotspots"
hotspot_cache = TTLCache()

FetchText = Callable[[str, float], str]

_CONFIDENCE = {
    "high": "high",
    "h": "high",
    "nominal": "nominal",
    "n": "nominal",
    "low": "low",
    "l": "low",
}


class FirmsConfigError(Exception):
    """Missing or rejected FIRMS map key. Safe to show to clients."""


class FirmsUpstreamError(Exception):
    """All VIIRS requests failed. Safe to show to clients."""


class InvalidMapKey(Exception):
    pass


class UnexpectedFirmsPayload(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_text(url: str, timeout: float) -> str:
    headers = {"User-Agent": "bc-wildfire-watch/0.1", "Accept": "text/csv"}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        response = client.get(url)
        # A rejected map key is often a 4xx body, not a 200 CSV. Do not log the
        # URL: it contains the map key.
        if response.status_code >= 400:
            lowered = response.text.lower()
            if "invalid" in lowered and "key" in lowered:
                raise InvalidMapKey()
            response.raise_for_status()
        return response.text


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _acquired_at(date: str, time_hhmm: str) -> str | None:
    day = date.strip()
    clock = time_hhmm.strip().zfill(4)
    if len(day) < 10 or len(clock) != 4 or not clock.isdigit():
        return None
    try:
        datetime.strptime(day[:10], "%Y-%m-%d")
    except ValueError:
        return None
    hour = int(clock[:2])
    minute = int(clock[2:])
    if hour > 23 or minute > 59:
        return None
    return f"{day[:10]}T{clock[:2]}:{clock[2:]}:00Z"


def parse_firms_csv(text: str) -> list[dict]:
    sample = text.lstrip("\ufeff").strip()
    if not sample:
        return []
    lowered = sample.lower()
    first_line = sample.splitlines()[0].lower()
    if "latitude" not in first_line:
        if "invalid" in lowered and "key" in lowered:
            raise InvalidMapKey()
        raise UnexpectedFirmsPayload()

    reader = csv.DictReader(io.StringIO(sample))
    rows: list[dict] = []
    seen: set[tuple] = set()
    for row in reader:
        if not row:
            continue
        try:
            lat = float(row.get("latitude") or "")
            lon = float(row.get("longitude") or "")
        except ValueError:
            continue
        if not (BC_SOUTH <= lat <= BC_NORTH and BC_WEST <= lon <= BC_EAST):
            continue
        satellite = (row.get("satellite") or "").strip()
        acq_date = (row.get("acq_date") or "").strip()
        acq_time = (row.get("acq_time") or "").strip()
        dedupe_key = (round(lat, 5), round(lon, 5), acq_date, acq_time, satellite)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        confidence_raw = (row.get("confidence") or "").strip().lower()
        daynight = (row.get("daynight") or "").strip().upper()
        rows.append(
            {
                "latitude": lat,
                "longitude": lon,
                "frp": _optional_float(row.get("frp")),
                "brightness": _optional_float(row.get("bright_ti4")),
                "acquired_at": _acquired_at(acq_date, acq_time),
                "satellite": satellite,
                "confidence": _CONFIDENCE.get(confidence_raw, "unknown"),
                "daynight": daynight if daynight in {"D", "N"} else "",
            }
        )
    return rows


def cluster_hotspots(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        lat_cell = math.floor(row["latitude"] / CLUSTER_CELL_DEG) * CLUSTER_CELL_DEG
        lon_cell = math.floor(row["longitude"] / CLUSTER_CELL_DEG) * CLUSTER_CELL_DEG
        buckets[f"{lat_cell:.2f},{lon_cell:.2f}"].append(row)

    clusters: list[dict] = []
    for cluster_id, members in buckets.items():
        frps = [member["frp"] for member in members if member["frp"] is not None]
        brightness = [
            member["brightness"] for member in members if member["brightness"] is not None
        ]
        times = [member["acquired_at"] for member in members if member["acquired_at"]]
        confidence = {"high": 0, "nominal": 0, "low": 0, "unknown": 0}
        daynight = {"D": 0, "N": 0}
        satellites: set[str] = set()
        for member in members:
            confidence[member["confidence"]] = confidence.get(member["confidence"], 0) + 1
            if member["daynight"] in daynight:
                daynight[member["daynight"]] += 1
            if member["satellite"]:
                satellites.add(member["satellite"])
        lats = [member["latitude"] for member in members]
        lons = [member["longitude"] for member in members]
        clusters.append(
            {
                "id": cluster_id,
                "latitude": round(sum(lats) / len(lats), 5),
                "longitude": round(sum(lons) / len(lons), 5),
                "hotspot_count": len(members),
                "max_frp": round(max(frps), 2) if frps else None,
                "mean_frp": round(sum(frps) / len(frps), 2) if frps else None,
                "max_brightness": round(max(brightness), 2) if brightness else None,
                "latest_acquisition": max(times) if times else None,
                "confidence": confidence,
                "satellites": sorted(satellites),
                "daynight": daynight,
                "bbox": [
                    round(min(lons), 5),
                    round(min(lats), 5),
                    round(max(lons), 5),
                    round(max(lats), 5),
                ],
            }
        )

    clusters.sort(
        key=lambda item: (
            item["max_frp"] is not None,
            item["max_frp"] or 0,
            item["hotspot_count"],
        ),
        reverse=True,
    )
    return clusters


def _source_url(base_url: str, map_key: str, source: str, day_range: int) -> str:
    bbox = f"{BC_WEST:.2f},{BC_SOUTH:.2f},{BC_EAST:.2f},{BC_NORTH:.2f}"
    return f"{base_url}/api/area/csv/{map_key}/{source}/{bbox}/{day_range}"


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    """Drop identical detections returned by more than one VIIRS source."""
    seen: set[tuple] = set()
    unique: list[dict] = []
    for row in rows:
        key = (
            round(row["latitude"], 5),
            round(row["longitude"], 5),
            row.get("acquired_at"),
            row.get("satellite"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _fetch_all(fetcher: FetchText) -> tuple[list[dict], list[str], int]:
    settings = get_settings()
    rows: list[dict] = []
    warnings: list[str] = []
    for source in settings.firms_sources:
        url = _source_url(
            settings.firms_base_url,
            settings.firms_map_key,
            source,
            settings.firms_day_range,
        )
        try:
            text = fetcher(url, settings.request_timeout)
            rows.extend(parse_firms_csv(text))
        except InvalidMapKey as exc:
            raise FirmsConfigError(
                "NASA FIRMS rejected FIRMS_MAP_KEY. Check the key and try again."
            ) from exc
        except Exception:
            # httpx errors include the request URL, which contains the map key.
            log.warning("FIRMS request failed for %s", source)
            warnings.append(f"{source} is temporarily unavailable.")

    if not rows and warnings and len(warnings) == len(settings.firms_sources):
        raise FirmsUpstreamError(
            "NASA FIRMS could not be reached for the configured VIIRS sources. Try again shortly."
        )
    rows = _dedupe_rows(rows)
    return cluster_hotspots(rows), warnings, len(rows)


def load_hotspots(fetcher: FetchText | None = None) -> dict:
    if fetcher is None:
        fetcher = fetch_text
    settings = get_settings()
    if not settings.firms_map_key:
        raise FirmsConfigError(
            "FIRMS_MAP_KEY is not configured. Add a NASA FIRMS map key to the API "
            "environment to load hotspots."
        )

    cached = hotspot_cache.get(CACHE_KEY)
    if cached is not None:
        return {**cached, "cached": True}

    clusters, warnings, hotspot_count = _fetch_all(fetcher)
    payload = {
        "region": "BC",
        "bbox": {
            "west": BC_WEST,
            "south": BC_SOUTH,
            "east": BC_EAST,
            "north": BC_NORTH,
        },
        "sources": list(settings.firms_sources),
        "day_range": settings.firms_day_range,
        "fetched_at": utc_now(),
        "hotspot_count": hotspot_count,
        "cluster_count": len(clusters),
        "clusters": clusters,
        "warnings": warnings,
    }
    hotspot_cache.set(CACHE_KEY, payload, settings.hotspot_cache_seconds)
    return {**payload, "cached": False}
