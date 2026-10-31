from fastapi import APIRouter, Query
from api.services import cwfis

router = APIRouter()


@router.get("/fwi")
async def current_fwi():
    """Current CWFIS Fire Weather Index station readings for BC."""
    return await cwfis.get_current_fwi()


@router.get("/nearest")
async def nearest_fwi(lat: float = Query(...), lon: float = Query(...)):
    """Nearest reporting CWFIS station within 100 km, or null."""
    return await cwfis.nearest_station(lat, lon)
