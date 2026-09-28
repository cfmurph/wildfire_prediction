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

# Suomi NPP VIIRS is being retired November 1, 2026.
# Real-time feed now uses NOAA-20 (J1) and NOAA-21 (J2).
# Historical archive (2012-2023 training) still uses SNPP.
FIRMS_NRT_SOURCES = [
    "VIIRS_NOAA21_NRT",   # NOAA-21 (J2, 2022+) — highest priority
    "VIIRS_NOAA20_NRT",   # NOAA-20 (J1, 2018+)
]
FIRMS_ARCHIVE_SOURCES = [
    "VIIRS_SNPP_NRT_2",   # SNPP archive (2012-2021)
    "VIIRS_NOAA20_NRT_2", # NOAA-20 archive (2018+)
    "VIIRS_NOAA21_NRT_2", # NOAA-21 archive (2022+)
]

# WMS tile fallback (no auth needed) — update to NOAA-20 layer
FIRMS_WMS = (
    "https://firms.modaps.eosdis.nasa.gov/mapserver/wms/fires/"
    "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
    "&LAYERS=fires_viirs_noaa20&STYLES=&FORMAT=image/png"
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

    # Try NOAA-21 → NOAA-20 in sequence; merge results for maximum coverage
    all_dfs = []
    for source in FIRMS_NRT_SOURCES:
        url = f"{FIRMS_BASE}/{map_key}/{source}/{BC_BBOX}/{days}"
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(url)
                if resp.ok:
                    partial = pd.read_csv(io.StringIO(resp.text))
                    if not partial.empty:
                        partial["satellite_source"] = source
                        all_dfs.append(partial)
        except Exception as exc:
            log.warning(f"FIRMS {source} failed: {exc}")

    try:
        df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    except Exception as exc:
        log.error(f"FIRMS merge failed: {exc}")
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
