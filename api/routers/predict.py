import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

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
