"""
Wildfire Prediction API
========================
FastAPI backend serving all four map views:
  /api/v1/fires/active   — live BC active fires (ArcGIS public feed)
  /api/v1/fires/hotspots — NASA FIRMS VIIRS NRT hotspots
  /api/v1/fires/history  — BC historical perimeters by year
  /api/v1/weather/fwi    — current CWFIS fire weather
  /api/v1/predict/spread — next-day spread probability (heuristic → ML)
  /api/v1/ignition/causes — where fires typically start, by month and cause

Deployed on Railway. All secrets are environment variables.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import fires, weather, predict, ignition

app = FastAPI(
    title="Wildfire Prediction API",
    description="Canadian wildfire spread prediction platform — BC Phase 1",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(fires.router,     prefix="/api/v1/fires",     tags=["fires"])
app.include_router(weather.router,   prefix="/api/v1/weather",   tags=["weather"])
app.include_router(predict.router,   prefix="/api/v1/predict",   tags=["predict"])
app.include_router(ignition.router,  prefix="/api/v1/ignition",  tags=["ignition"])


@app.get("/")
async def root():
    return {
        "name": "wildfire-prediction-api",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
