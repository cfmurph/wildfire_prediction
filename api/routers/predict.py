from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

# Written daily by scripts/predict_spread.py
SPREAD_MAP = Path("data/live/spread.geojson")


@router.get("/spread-map")
async def spread_map():
    """Where active fires are most likely to grow next, as GeoJSON areas banded "highest" and "elevated"."""
    if not SPREAD_MAP.exists():
        raise HTTPException(503, "No spread prediction has been run yet")
    return FileResponse(SPREAD_MAP, media_type="application/geo+json")
