import json
from pathlib import Path
from fastapi import APIRouter, Query
from api.services import predictor

router = APIRouter()

# Pre-load climatology at import time (151 KB — negligible)
_CLIM_PATH = Path("data/processed/stats/fwi_climatology.json")
_CLIMATOLOGY: dict | None = None


def _load_climatology() -> dict | None:
    global _CLIMATOLOGY
    if _CLIMATOLOGY is None and _CLIM_PATH.exists():
        with open(_CLIM_PATH) as f:
            _CLIMATOLOGY = json.load(f)
    return _CLIMATOLOGY


def _clim_to_geojson(month_data: dict) -> dict:
    """Convert flat-list climatology record to GeoJSON FeatureCollection."""
    lats = month_data["lat"]
    lons = month_data["lon"]
    fwis = month_data["fwi_mean"]
    probs = month_data["burn_prob"]

    def _risk_class(p: float) -> str:
        if p < 0.03: return "Low"
        if p < 0.08: return "Moderate"
        if p < 0.18: return "High"
        return "Very High"

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lons[i], lats[i]]},
            "properties": {
                "burn_probability": probs[i],
                "fwi_mean": fwis[i],
                "risk_class": _risk_class(probs[i]),
            },
        }
        for i in range(len(lats))
    ]
    return {"type": "FeatureCollection", "features": features}


@router.get("/map")
async def risk_map():
    """
    Long-term burn likelihood grid for BC — current month climatology.

    Phase 1: Historical burn frequency climatology.
    Phase 2: BurnP3+ simulated burn probability surfaces.
    """
    return predictor.risk_grid()


@router.get("/monthly")
async def risk_monthly(
    month: int = Query(default=7, ge=1, le=12, description="Month (1=Jan, 12=Dec)"),
):
    """
    Monthly FWI climatology risk surface for BC.

    Returns a GeoJSON FeatureCollection with burn probability for each grid cell,
    based on historical ERA5-HRS FWI means for the requested month.

    Used to power the 12-month forward-looking risk calendar in the web UI.
    """
    clim = _load_climatology()
    if clim and str(month) in clim:
        return _clim_to_geojson(clim[str(month)])
    # Fallback to synthetic grid if climatology not available
    return predictor.risk_grid()


@router.get("/calendar")
async def risk_calendar():
    """
    Full 12-month risk calendar — summary statistics per month for BC.

    Returns a list of {month, mean_burn_prob, peak_month, fire_season} records
    for the risk timeline bar in the UI.
    """
    clim = _load_climatology()
    months = []
    for m in range(1, 13):
        key = str(m)
        if clim and key in clim:
            probs = clim[key]["burn_prob"]
            mean_p = sum(probs) / max(len(probs), 1)
            max_p = max(probs) if probs else 0.0
        else:
            # Synthetic seasonal pattern
            seasonal = [0.01, 0.01, 0.01, 0.03, 0.05, 0.08, 0.18, 0.16, 0.06, 0.03, 0.01, 0.01]
            mean_p = max_p = seasonal[m - 1]

        months.append({
            "month": m,
            "month_name": ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"][m - 1],
            "mean_burn_prob": round(mean_p, 4),
            "peak_burn_prob": round(max_p, 4),
            "fire_season": m in [5, 6, 7, 8, 9],   # May–Sep
        })
    return {"months": months}
