import time
from collections import deque
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from api.services import grok, predictor

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
    fwi: float = 20.0
    isi: float = 8.0
    wind_speed_ms: float = 5.0
    wind_dir_deg: float = 270.0


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

    # Get spread forecast
    spread = predictor.spread_forecast(
        lat=req.lat,
        lon=req.lon,
        wind_dir_deg=req.wind_dir_deg,
        wind_speed_ms=req.wind_speed_ms,
        fwi=req.fwi,
        isi=req.isi,
        hours=24.0,
    )

    # Extract p50 and p75 area estimates
    p50_ha = next(
        (f["properties"]["area_ha"] for f in spread["features"]
         if f["properties"]["percentile"] == "p50"), 0.0
    )
    p75_ha = next(
        (f["properties"]["area_ha"] for f in spread["features"]
         if f["properties"]["percentile"] == "p75"), 0.0
    )

    report = await grok.generate_situation_report(
        fire_id=req.fire_id,
        fire_name=req.fire_name,
        current_area_ha=req.current_area_ha,
        lat=req.lat,
        lon=req.lon,
        fwi=req.fwi,
        isi=req.isi,
        wind_speed_ms=req.wind_speed_ms,
        wind_dir_deg=req.wind_dir_deg,
        spread_p50_ha=p50_ha,
        spread_p75_ha=p75_ha,
    )

    return SituationResponse(report=report, spread_geojson=spread)
