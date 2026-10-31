from datetime import date

from fastapi import APIRouter, HTTPException, Query
from api.services import firms, bc_fires

router = APIRouter()


@router.get("/active")
async def active_fires():
    """Current active BC wildfires from BC Wildfire Service ArcGIS."""
    return _or_502(await bc_fires.get_active_fires())


@router.get("/hotspots")
async def hotspots(days: int = Query(default=1, ge=1, le=10)):
    """NASA FIRMS VIIRS NRT hotspots for BC (requires FIRMS_MAP_KEY)."""
    return await firms.get_hotspots_geojson(days=days)


@router.get("/history")
async def history(year: int = Query(default=2023, ge=2006)):
    """BC fire perimeters for a given year, including the current season."""
    if year > date.today().year:
        raise HTTPException(422, "year is in the future")
    return _or_502(await bc_fires.get_historical_perimeters(year=year))


def _or_502(data: dict | None) -> dict:
    """An upstream failure is an error, so the map never shows it as "no fires"."""
    if data is None:
        raise HTTPException(502, "BC Wildfire Service data is unavailable")
    return data
