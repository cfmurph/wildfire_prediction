"""
CWFIS Fire Weather Index service.

Fetches current FWI system values for BC weather stations from the
Canadian Wildland Fire Information System public API.

Fallback: returns a synthetic grid of BC FWI values based on typical
fire season climatology when the live API is unavailable.
"""

from __future__ import annotations

import logging
import math
from datetime import date

import httpx

log = logging.getLogger(__name__)

# CWFIS current weather station data (public, no auth)
CWFIS_STATIONS_URL = (
    "https://cwfis.cfs.nrcan.gc.ca/fireweather/beta/stns"
    "?lat1=48&lat2=60&lon1=-139&lon2=-114"
    "&yr={yr}&mo={mo}&day={da}"
)

# BC bounding box for synthetic fallback
BC_LAT_RANGE = (49.0, 59.0)
BC_LON_RANGE = (-139.0, -115.0)


async def get_current_fwi() -> dict:
    """
    Fetch current CWFIS FWI station readings as GeoJSON.
    Falls back to a synthetic grid if the API is unavailable.
    """
    today = date.today()
    url = CWFIS_STATIONS_URL.format(yr=today.year, mo=today.month, da=today.day)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        stations = data if isinstance(data, list) else data.get("stations", [])
        features = []
        for stn in stations:
            lon = float(stn.get("lon") or stn.get("longitude") or 0)
            lat = float(stn.get("lat") or stn.get("latitude") or 0)
            if not (-139 <= lon <= -114 and 48 <= lat <= 60):
                continue
            fwi = float(stn.get("fwi") or 0)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "station_id":   str(stn.get("stn_id") or stn.get("id", "")),
                    "station_name": str(stn.get("stn_name") or stn.get("name", "")),
                    "FFMC": float(stn.get("ffmc") or 0),
                    "DMC":  float(stn.get("dmc") or 0),
                    "DC":   float(stn.get("dc") or 0),
                    "ISI":  float(stn.get("isi") or 0),
                    "BUI":  float(stn.get("bui") or 0),
                    "FWI":  fwi,
                    "fwi_class": _fwi_class(fwi),
                    "wind_speed": float(stn.get("ws") or 0),
                    "wind_dir":   float(stn.get("wd") or 0),
                    "temp":       float(stn.get("temp") or 0),
                    "rh":         float(stn.get("rh") or 0),
                    "precip":     float(stn.get("prec") or 0),
                },
            })

        if features:
            return {"type": "FeatureCollection", "features": features}

    except Exception as exc:
        log.warning("CWFIS live API unavailable (%s) — using synthetic grid", type(exc).__name__)

    return _synthetic_fwi_grid()


def _fwi_class(fwi: float) -> str:
    if fwi < 5:   return "Low"
    if fwi < 12:  return "Moderate"
    if fwi < 20:  return "High"
    if fwi < 30:  return "Very High"
    return "Extreme"


def _synthetic_fwi_grid() -> dict:
    """
    Generate a 10×10 synthetic FWI grid over BC for display when live data
    is unavailable. Values reflect typical mid-summer conditions.
    Highest FWI in the BC Interior (historically most fire-prone).
    """
    features = []
    lat_min, lat_max = BC_LAT_RANGE
    lon_min, lon_max = BC_LON_RANGE

    for i in range(10):
        for j in range(10):
            lat = lat_min + (lat_max - lat_min) * (i / 9)
            lon = lon_min + (lon_max - lon_min) * (j / 9)
            # Interior BC has higher FWI — use distance from coast proxy
            coast_dist = (lon + 130) / 10   # 0 near coast, 1 in NE interior
            lat_factor = 1 - abs(lat - 51) / 10  # peaks around 51°N
            fwi = max(0, 18 * coast_dist * lat_factor + 5 + (i % 3) * 2)

            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 2), round(lat, 2)]},
                "properties": {
                    "station_id": f"synthetic_{i}_{j}",
                    "station_name": "Synthetic",
                    "FWI": round(fwi, 1),
                    "fwi_class": _fwi_class(fwi),
                    "FFMC": round(min(101, 70 + fwi * 0.8), 1),
                    "DMC":  round(fwi * 2.5, 1),
                    "DC":   round(fwi * 10, 1),
                    "ISI":  round(fwi * 0.5, 1),
                    "BUI":  round(fwi * 2.8, 1),
                    "wind_speed": 20 + (i % 4) * 5,
                    "wind_dir": (j * 36) % 360,
                    "temp": 25 + (i % 5),
                    "rh": 20 - (j % 5),
                    "precip": 0.0,
                    "is_synthetic": True,
                },
            })

    return {"type": "FeatureCollection", "features": features}
