"""
CWFIS Fire Weather Index service.

Fetches the latest FWI system values for BC-area weather stations from the
Canadian Wildland Fire Information System GeoServer (public, no auth).

Returns an empty FeatureCollection when the live API is unavailable. Outside
the fire season few or no BC stations report FWI.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import httpx

from api.services.cache import ttl_cache

log = logging.getLogger(__name__)

# Latest daily report per station, Canada-wide; filtered to BC below
CWFIS_STATIONS_URL = (
    "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wfs"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeNames=public:firewx_stns_current&outputFormat=application/json&srsName=EPSG:4326"
)


@ttl_cache(300)
async def get_current_fwi() -> dict:
    """
    Fetch current CWFIS FWI station readings as GeoJSON.
    Returns no stations if the API is unavailable.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(CWFIS_STATIONS_URL)
            resp.raise_for_status()
            data = resp.json()

        features = []
        for feat in data.get("features") or []:
            stn = feat.get("properties") or {}
            lon, lat, fwi = stn.get("lon"), stn.get("lat"), stn.get("fwi")
            # Stations that report weather but no FWI have null indices
            if lon is None or lat is None or fwi is None:
                continue
            if not (-139 <= lon <= -114 and 48 <= lat <= 60):
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "station_id":   str(stn.get("aes") or stn.get("wmo") or ""),
                    "station_name": str(stn.get("name") or "").strip(),
                    "report_date":  stn.get("rep_date"),
                    "FFMC": float(stn.get("ffmc") or 0),
                    "DMC":  float(stn.get("dmc") or 0),
                    "DC":   float(stn.get("dc") or 0),
                    "ISI":  float(stn.get("isi") or 0),
                    "BUI":  float(stn.get("bui") or 0),
                    "FWI":  float(fwi),
                    "fwi_class": _fwi_class(float(fwi)),
                    "wind_speed": float(stn.get("ws") or 0),   # km/h
                    "wind_dir":   float(stn.get("wdir") or 0),
                    "temp":       float(stn.get("temp") or 0),
                    "rh":         float(stn.get("rh") or 0),
                    "precip":     float(stn.get("precip") or 0),
                },
            })
        return {"type": "FeatureCollection", "features": features}

    except Exception as exc:
        log.warning("CWFIS live API unavailable (%s)", type(exc).__name__)

    return {"type": "FeatureCollection", "features": []}


async def nearest_station(lat: float, lon: float, max_km: float = 100.0) -> Optional[dict]:
    """Properties of the nearest reporting BC station within max_km, or None."""
    best, best_km = None, max_km
    for feat in (await get_current_fwi())["features"]:
        s_lon, s_lat = feat["geometry"]["coordinates"]
        # Equirectangular distance is accurate enough at 100 km
        x = math.radians(s_lon - lon) * math.cos(math.radians((s_lat + lat) / 2))
        km = 6371 * math.hypot(x, math.radians(s_lat - lat))
        if km <= best_km:
            best, best_km = {**feat["properties"], "distance_km": round(km, 1)}, km
    return best


def _fwi_class(fwi: float) -> str:
    if fwi < 5:   return "Low"
    if fwi < 12:  return "Moderate"
    if fwi < 20:  return "High"
    if fwi < 30:  return "Very High"
    return "Extreme"
