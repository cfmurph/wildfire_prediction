import time
from collections import deque
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from api.services import cwfis, grok, predictor

router = APIRouter()

# Each report is a billed Grok call, so cap them across all users (per process).
MAX_REPORTS_PER_HOUR = 60
_recent_reports: deque[float] = deque()


class SituationRequest(BaseModel):
    fire_id: str
    fire_name: Optional[str] = None
    lat: float
    lon: float
    current_area_ha: float = 0.0
    # Real fires use the nearest reporting CWFIS station, looked up here.
    # Client weather is used only for hypothetical fires and is labelled as assumed.
    hypothetical: bool = False
    fwi: Optional[float] = None
    isi: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_dir_deg: Optional[float] = None


class SituationResponse(BaseModel):
    report: str
    spread_geojson: dict


@router.post("", response_model=SituationResponse)
async def situation_report(req: SituationRequest):
    """
    Generate a Grok AI situation report + spread forecast for a fire.
    Combines the spread prediction with an NL briefing for emergency managers.
    """
    now = time.monotonic()
    while _recent_reports and now - _recent_reports[0] > 3600:
        _recent_reports.popleft()
    if len(_recent_reports) >= MAX_REPORTS_PER_HOUR:
        raise HTTPException(429, "Situation report limit reached. Try again later.")
    _recent_reports.append(now)

    if req.hypothetical:
        weather = (req.fwi, req.isi, req.wind_speed_ms, req.wind_dir_deg)
        weather_source = "assumed typical weather, not a measurement"
    else:
        stn = await cwfis.nearest_station(req.lat, req.lon) or {}
        weather = (stn.get("FWI"), stn.get("ISI"), stn.get("wind_speed"), stn.get("wind_dir"))
        if stn:
            weather = (*weather[:2], weather[2] / 3.6, weather[3])   # CWFIS reports km/h
        weather_source = f"CWFIS station {stn.get('station_name')}, {stn.get('distance_km')} km away"
    fwi, isi, wind_speed_ms, wind_dir_deg = weather
    spread = {"type": "FeatureCollection", "features": []}
    p50_ha = p75_ha = None
    if None not in weather:
        spread = predictor.spread_forecast(
            lat=req.lat,
            lon=req.lon,
            wind_dir_deg=wind_dir_deg,
            wind_speed_ms=wind_speed_ms,
            fwi=fwi,
            isi=isi,
            hours=24.0,
        )
        areas = {f["properties"]["percentile"]: f["properties"]["area_ha"] for f in spread["features"]}
        p50_ha, p75_ha = areas.get("p50", 0.0), areas.get("p75", 0.0)

    report = await grok.generate_situation_report(
        fire_id=req.fire_id,
        fire_name=req.fire_name,
        current_area_ha=req.current_area_ha,
        lat=req.lat,
        lon=req.lon,
        fwi=fwi,
        isi=isi,
        wind_speed_ms=wind_speed_ms,
        wind_dir_deg=wind_dir_deg,
        weather_source=weather_source,
        spread_p50_ha=p50_ha,
        spread_p75_ha=p75_ha,
    )

    return SituationResponse(report=report, spread_geojson=spread)
