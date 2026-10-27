"""
CWFIS Fire Weather Index service.

Fetches current FWI system values for BC weather stations from the
Canadian Wildland Fire Information System public API.

Returns an empty FeatureCollection when the live API is unavailable.
"""

from __future__ import annotations

import logging
from datetime import date

import httpx

log = logging.getLogger(__name__)

# CWFIS current weather station data (public, no auth)
CWFIS_STATIONS_URL = (
    "https://cwfis.cfs.nrcan.gc.ca/fireweather/beta/stns"
    "?lat1=48&lat2=60&lon1=-139&lon2=-114"
    "&yr={yr}&mo={mo}&day={da}"
)


async def get_current_fwi() -> dict:
    """
    Fetch current CWFIS FWI station readings as GeoJSON.
    Returns no stations if the API is unavailable.
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
        log.warning("CWFIS live API unavailable (%s)", type(exc).__name__)

    return {"type": "FeatureCollection", "features": []}


def _fwi_class(fwi: float) -> str:
    if fwi < 5:   return "Low"
    if fwi < 12:  return "Moderate"
    if fwi < 20:  return "High"
    if fwi < 30:  return "Very High"
    return "Extreme"

