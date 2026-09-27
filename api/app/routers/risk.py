"""Long-term burn likelihood. Add the risk layer in this module later."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.routers.planned import coming_soon

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("")
def risk_index() -> JSONResponse:
    return coming_soon("risk")
