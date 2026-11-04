import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from api.services import spread_store

log = logging.getLogger(__name__)
router = APIRouter()

# Written daily by scripts/predict_spread.py; serves the map only when DATABASE_URL is not set
SPREAD_MAP = Path("data/live/spread.geojson")
MAX_AGE_SECONDS = 2 * 24 * 3600  # older means the daily run has stopped: show nothing rather than old bands
# A run predicts from the previous UTC day, so its date is normally one or two days back
MAX_AGE_DAYS = 3


@router.get("/spread-map")
def spread_map():
    """Where active fires are most likely to grow next, as GeoJSON areas banded "highest" and "elevated"."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return _from_file()
    try:
        latest = spread_store.latest_run(database_url)
    except psycopg.OperationalError as exc:
        log.error("Spread database unavailable (%s)", type(exc).__name__)
        raise HTTPException(503, "The spread prediction is unavailable") from exc
    if latest is None:
        raise HTTPException(503, "No spread prediction has been run yet")
    day, geojson = latest
    if (datetime.now(timezone.utc).date() - day).days > MAX_AGE_DAYS:
        raise HTTPException(503, "The latest spread prediction is out of date")
    return Response(geojson, media_type="application/geo+json")


def _from_file():
    if not SPREAD_MAP.exists():
        raise HTTPException(503, "No spread prediction has been run yet")
    if time.time() - SPREAD_MAP.stat().st_mtime > MAX_AGE_SECONDS:
        raise HTTPException(503, "The latest spread prediction is out of date")
    return FileResponse(SPREAD_MAP, media_type="application/geo+json")
