"""FastAPI app: BC hotspot map data and Grok situation reports."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.firms import FirmsConfigError, FirmsUpstreamError, load_hotspots
from app.reporting import MissingGrokKey, ReportFailed, UnknownCluster, generate_report
from app.schemas import ReportRequest
from app.weather import load_weather

logging.basicConfig(level=logging.INFO)
# httpx logs full request URLs at INFO, which would include the FIRMS map key.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI(
    title="BC Wildfire Watch",
    version="0.1.0",
    summary="Near-real-time VIIRS hotspots in British Columbia and Grok situation reports.",
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_origins) if _settings.cors_origins != ("*",) else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "region": "BC",
        "firms_configured": bool(settings.firms_map_key),
        "grok_configured": bool(settings.xai_api_key),
    }


@app.get("/hotspots")
def hotspots() -> dict:
    try:
        return load_hotspots()
    except FirmsConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FirmsUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/weather")
def weather() -> dict:
    return load_weather()


@app.post("/report")
def report(body: ReportRequest) -> dict:
    try:
        return generate_report(body.cluster.model_dump())
    except MissingGrokKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UnknownCluster as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReportFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
