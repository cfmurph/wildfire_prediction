"""
NASA FIRMS VIIRS NRT hotspot service.

With MAP_KEY: returns point GeoJSON with fire attributes (FRP, brightness, confidence).
Without MAP_KEY: returns empty FeatureCollection — WMS tile fallback is used on frontend.

BC bounding box: lon -139 to -114, lat 48 to 60
"""

from __future__ import annotations

import io
import os
import logging
from typing import Optional

import httpx
import pandas as pd

log = logging.getLogger(__name__)

BC_BBOX = "-139,48,-114,60"
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
        log.error(f"FIRMS fetch failed: {exc}")
        return _empty_fc()

    if df.empty:
        return _empty_fc()

    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row.get("longitude", 0)), float(row.get("latitude", 0))],
            },
            "properties": {
                "bright_ti4": float(row.get("bright_ti4", 0)),
                "bright_ti5": float(row.get("bright_ti5", 0)),
                "frp": float(row.get("frp", 0)),
                "confidence": str(row.get("confidence", "")),
                "acq_date": str(row.get("acq_date", "")),
                "acq_time": str(row.get("acq_time", "")),
                "satellite": str(row.get("satellite", "SNPP")),
                "daynight": str(row.get("daynight", "")),
            },
        })

    return {"type": "FeatureCollection", "features": features}


def _empty_fc() -> dict:
    return {"type": "FeatureCollection", "features": []}
