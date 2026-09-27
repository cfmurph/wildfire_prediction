from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from api.services import grok, predictor

router = APIRouter()


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
