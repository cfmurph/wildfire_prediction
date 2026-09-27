"""Next-day spread per fire. Add model-backed predictions in this module later."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.routers.planned import coming_soon

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("")
def predict_index() -> JSONResponse:
    return coming_soon("predict")
