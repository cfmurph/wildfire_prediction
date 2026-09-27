from fastapi import APIRouter, Query
from api.services import firms, bc_fires

router = APIRouter()


@router.get("/active")
async def active_fires():
    """Current active BC wildfires from BC Wildfire Service ArcGIS."""
    return await bc_fires.get_active_fires()


@router.get("/hotspots")
async def hotspots(days: int = Query(default=1, ge=1, le=10)):
    """NASA FIRMS VIIRS NRT hotspots for BC (requires FIRMS_MAP_KEY)."""
    return await firms.get_hotspots_geojson(days=days)


@router.get("/history")
async def history(year: int = Query(default=2023, ge=2006, le=2024)):
    """BC historical fire perimeters for a given year."""
    return await bc_fires.get_historical_perimeters(year=year)
