import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from api.services import predictor

router = APIRouter()

# Written daily by scripts/predict_spread.py
SPREAD_MAP = Path("data/live/spread.geojson")
MAX_AGE_SECONDS = 2 * 24 * 3600  # older means the daily run has stopped: show nothing rather than old bands


@router.get("/spread-map")
async def spread_map():
    """Where active fires are most likely to grow next, as GeoJSON areas banded "highest" and "elevated"."""
    if not SPREAD_MAP.exists():
        raise HTTPException(503, "No spread prediction has been run yet")
    if time.time() - SPREAD_MAP.stat().st_mtime > MAX_AGE_SECONDS:
        raise HTTPException(503, "The latest spread prediction is out of date")
    return FileResponse(SPREAD_MAP, media_type="application/geo+json")


@router.get("/spread")
async def spread_prediction(
    lat: float = Query(..., description="Fire centroid latitude"),
    lon: float = Query(..., description="Fire centroid longitude"),
    wind_dir: float = Query(default=270.0, description="Wind direction (degrees, FROM)"),
    wind_speed: float = Query(default=5.0, description="Wind speed (m/s)"),
    fwi: float = Query(default=20.0, description="Fire Weather Index"),
    isi: float = Query(default=8.0, description="Initial Spread Index"),
    hours: float = Query(default=24.0, description="Prediction horizon (hours)"),
):
    """
    Next-day fire spread probability forecast.

    Returns a GeoJSON FeatureCollection with three spread polygons:
    p25 (conservative), p50 (median), p75 (upper bound).

    Currently uses a physics-inspired heuristic model.
    Replaces with trained ConvLSTM once model training completes.
    """
    return predictor.spread_forecast(
        lat=lat, lon=lon,
        wind_dir_deg=wind_dir,
        wind_speed_ms=wind_speed,
        fwi=fwi,
        isi=isi,
        hours=hours,
    )
