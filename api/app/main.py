"""FastAPI app: one map, four views. Only the current-wildfire view is implemented."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import current, history, predict, risk
from app.routers.planned import PLANNED_VIEWS

logging.basicConfig(level=logging.INFO)
# httpx logs full request URLs at INFO, which would include the FIRMS map key.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(
    title="BC Wildfire Watch",
    version="0.1.0",
    summary=(
        "British Columbia wildfire map. Current hotspots are live; "
        "history, risk, and spread are planned."
    ),
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_origins) if _settings.cors_origins != ("*",) else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(current.router)
app.include_router(history.router)
app.include_router(risk.router)
app.include_router(predict.router)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "region": "BC",
        "firms_configured": bool(settings.firms_map_key),
        "grok_configured": bool(settings.xai_api_key),
        "views": {
            "history": "coming_soon",
            "risk": "coming_soon",
            "current": "available",
            "predict": "coming_soon",
        },
        "planned": PLANNED_VIEWS,
    }
