"""
Where fires typically start in a given month, from recorded BC fire starts.

scripts/compute_fire_starts.py builds the density grids; this serves them.
"""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

FIRE_STARTS = Path("data/processed/stats/fire_starts.json")


@lru_cache(maxsize=1)
def _fire_starts() -> dict | None:
    return json.loads(FIRE_STARTS.read_text()) if FIRE_STARTS.exists() else None


@router.get("/causes")
async def ignition_causes(month: int = Query(default=7, ge=1, le=12)):
    """Lightning- and human-caused fire-start density for the month, as GeoJSON points."""
    data = _fire_starts()
    if data is None:
        raise HTTPException(503, "Fire-start data is not available")

    def collection(cells: list) -> dict:
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
             "properties": {"starts": density}}
            for lon, lat, density in cells
        ]}

    by_cause = data["months"][str(month)]
    return {
        "lightning": collection(by_cause["lightning"]),
        "human": collection(by_cause["human"]),
        "unit": data["unit"],
        "years": data["years"],
    }
