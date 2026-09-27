"""Past BC fires, 2012–2023. Add the time-slider queries in this module later."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.routers.planned import coming_soon

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
def history_index() -> JSONResponse:
    return coming_soon("history")
