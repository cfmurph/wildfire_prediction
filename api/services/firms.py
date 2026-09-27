"""
NASA FIRMS VIIRS NRT hotspot service.

With MAP_KEY: returns point GeoJSON with fire attributes (FRP, brightness, confidence).
Without MAP_KEY: returns empty FeatureCollection — WMS tile fallback is used on frontend.

BC bounding box: lon -139 to -114, lat 48 to 60
"""

from __future__ import annotations

import io
import math
import os
import logging

import httpx
import pandas as pd

log = logging.getLogger(__name__)

BC_BBOX = "-139,48,-114,60"
BC_LON_MIN, BC_LON_MAX = -139.0, -114.0
BC_LAT_MIN, BC_LAT_MAX = 48.0, 60.0
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_WMS = (
    "https://firms.modaps.eosdis.nasa.gov/mapserver/wms/fires/"
    "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
    "&LAYERS=fires_viirs_snpp&STYLES=&FORMAT=image/png"
    "&TRANSPARENT=true&BGCOLOR=0x000000&CRS=EPSG:4326"
    "&WIDTH=256&HEIGHT=256"
)


async def get_hotspots_geojson(days: int = 1) -> dict:
    """
    Fetch VIIRS NRT hotspots for BC as GeoJSON FeatureCollection.

    Falls back to empty collection if MAP_KEY is not set.
    """
    map_key = os.environ.get("FIRMS_MAP_KEY", "")
    if not map_key:
        log.warning("FIRMS_MAP_KEY not set — returning empty hotspot collection")
        return _empty_fc()

    url = f"{FIRMS_BASE}/{map_key}/VIIRS_SNPP_NRT/{BC_BBOX}/{days}"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        # httpx errors include the request URL, which contains the map key.
        log.error("FIRMS fetch failed (%s)", type(exc).__name__)
        return _empty_fc()

    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return _empty_fc()

    features = []
    for _, row in df.iterrows():
        lat = _finite(row.get("latitude"))
        lon = _finite(row.get("longitude"))
        if lat is None or lon is None:
            continue
        if not (BC_LAT_MIN <= lat <= BC_LAT_MAX and BC_LON_MIN <= lon <= BC_LON_MAX):
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "bright_ti4": _or_zero(row.get("bright_ti4")),
                "bright_ti5": _or_zero(row.get("bright_ti5")),
                "frp": _or_zero(row.get("frp")),
                "confidence": str(row.get("confidence", "")),
                "acq_date": str(row.get("acq_date", "")),
                "acq_time": str(row.get("acq_time", "")),
                "satellite": str(row.get("satellite", "SNPP")),
                "daynight": str(row.get("daynight", "")),
            },
        })

    return {"type": "FeatureCollection", "features": features}


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _or_zero(value: object) -> float:
    number = _finite(value)
    return 0.0 if number is None else number


def _empty_fc() -> dict:
    return {"type": "FeatureCollection", "features": []}
