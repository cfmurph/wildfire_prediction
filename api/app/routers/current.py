"""Current wildfires: VIIRS hotspots, CWFIS stations, and Grok situation reports."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.firms import FirmsConfigError, FirmsUpstreamError, load_hotspots
from app.reporting import MissingGrokKey, ReportFailed, UnknownCluster, generate_report
from app.schemas import ReportRequest
from app.weather import load_weather

router = APIRouter(tags=["current"])


@router.get("/hotspots")
def hotspots() -> dict:
    try:
        return load_hotspots()
    except FirmsConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FirmsUpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/weather")
def weather() -> dict:
    return load_weather()


@router.post("/report")
def report(body: ReportRequest) -> dict:
    try:
        return generate_report(body.cluster.model_dump())
    except MissingGrokKey as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UnknownCluster as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReportFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
