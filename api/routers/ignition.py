"""
Fire Ignition Prediction API
Predicts P(fire starts here this month) per km² grid cell across BC.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from fastapi import APIRouter, Query

log = logging.getLogger(__name__)
router = APIRouter()

_MODEL = None
_CLIM = None
_DENSITIES = None
_MONTHLY_RATES = None


def _load_resources():
    global _MODEL, _CLIM, _DENSITIES, _MONTHLY_RATES
    if _MODEL is not None:
        return _MODEL, _CLIM, _DENSITIES, _MONTHLY_RATES

    import joblib

    model_path  = Path("experiments/checkpoints/ignition_rf.joblib")
    clim_path   = Path("data/processed/stats/fwi_climatology.json")
    density_path = Path("data/processed/stats/ignition_densities.json")
    rates_path  = Path("data/processed/stats/monthly_fire_rates.json")

    if model_path.exists():
        _MODEL = joblib.load(model_path)
        log.info("Ignition RF model loaded")

    if clim_path.exists():
        with open(clim_path) as f:
            _CLIM = json.load(f)

    if density_path.exists():
        with open(density_path) as f:
            raw = json.load(f)
        _DENSITIES = {
            "lightning": {tuple(map(float, k.split(","))): v for k, v in raw["lightning"].items()},
            "human":     {tuple(map(float, k.split(","))): v for k, v in raw["human"].items()},
        }

    if rates_path.exists():
        with open(rates_path) as f:
            _MONTHLY_RATES = json.load(f)
        log.info("Monthly fire rates loaded")

    return _MODEL, _CLIM, _DENSITIES, _MONTHLY_RATES


@router.get("/monthly")
async def ignition_monthly(
    month: int = Query(default=7, ge=1, le=12, description="Month (1=Jan, 12=Dec)"),
    grid_step: float = Query(default=0.25, ge=0.1, le=1.0, description="Grid resolution in degrees"),
):
    """
    Predict fire ignition probability across BC for a given month.

    Returns GeoJSON FeatureCollection where each point is a ~1km² grid cell
    with burn_probability from the trained Random Forest ignition model.

    Features used: lightning density, human activity density, FWI climatology,
    location (coast vs interior gradient), and seasonality.
    """
    model, clim, densities = _load_resources()

    if model is None:
        # Return synthetic climatology as fallback
        from api.routers.risk import risk_monthly
        return await risk_monthly(month=month)

    from src.models.ignition_rf import predict_ignition_grid

    return predict_ignition_grid(
        model=model,
        month=month,
        fwi_climatology=clim,
        lightning_density=densities.get("lightning") if densities else None,
        human_density=densities.get("human") if densities else None,
        grid_step=grid_step,
    )


@router.get("/causes")
async def ignition_causes(
    month: int = Query(default=7, ge=1, le=12),
):
    """
    Return separate lightning-caused and human-caused ignition probability grids.
    Allows the UI to show two overlapping layers with different colours.
    """
    model, clim, densities = _load_resources()
    if model is None:
        return {"lightning": {"type": "FeatureCollection", "features": []},
                "human": {"type": "FeatureCollection", "features": []}}

    from src.models.ignition_rf import predict_ignition_grid, build_ignition_features, FEATURE_COLS, snap_density
    import numpy as np, pandas as pd
    BC_LAT_MIN, BC_LAT_MAX = 48.0, 60.0
    BC_LON_MIN, BC_LON_MAX = -139.1, -114.0
    grid_step = 0.25
    ld = densities.get("lightning", {}) if densities else {}
    hd = densities.get("human", {}) if densities else {}

    lats = np.arange(BC_LAT_MIN, BC_LAT_MAX, grid_step)
    lons = np.arange(BC_LON_MIN, BC_LON_MAX, grid_step)
    rows = []
    for lat in lats:
        for lon in lons:
            feat = build_ignition_features(float(lat), float(lon), month,
                                           snap_density(lat, lon, ld),
                                           snap_density(lat, lon, hd), clim)
            rows.append(feat)
    df = pd.DataFrame(rows)
    probs = model.predict_proba(df[FEATURE_COLS].values.astype(np.float32))[:, 1]

    lightning_feats, human_feats = [], []
    for i, (lat, lon, prob) in enumerate(zip(df["lat"], df["lon"], probs)):
        if prob < 0.01:
            continue
        ld_val = float(df.iloc[i]["lightning_density"])
        hd_val = float(df.iloc[i]["human_density"])
        total = max(ld_val + hd_val, 1e-6)
        lightning_share = ld_val / total
        human_share = hd_val / total
        base = {"type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(float(lon), 3), round(float(lat), 3)]}}
        if lightning_share > 0.1:
            lightning_feats.append({**base, "properties": {"prob": round(float(prob * lightning_share), 4)}})
        if human_share > 0.1:
            human_feats.append({**base, "properties": {"prob": round(float(prob * human_share), 4)}})

    return {
        "lightning": {"type": "FeatureCollection", "features": lightning_feats},
        "human":     {"type": "FeatureCollection", "features": human_feats},
    }
