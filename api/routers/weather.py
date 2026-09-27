from fastapi import APIRouter
from api.services import cwfis

router = APIRouter()


@router.get("/fwi")
async def current_fwi():
    """Current CWFIS Fire Weather Index station readings for BC."""
    return await cwfis.get_current_fwi()
