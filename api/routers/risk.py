from fastapi import APIRouter
from api.services import predictor

router = APIRouter()


@router.get("/map")
async def risk_map():
    """
    Long-term burn likelihood grid for BC.

    Phase 1: Historical burn frequency climatology.
    Phase 2: BurnP3+ simulated burn probability surfaces.
    """
    return predictor.risk_grid()
