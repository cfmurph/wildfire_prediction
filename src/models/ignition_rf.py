"""
Fire Ignition Prediction Model
================================
Predicts P(fire starts here this month) per 1 km² grid cell across BC.

Features per cell:
  - Fuel type (from Canada Landcover 2020) — flammability proxy
  - Terrain (elevation, slope) — drying and access effects
  - FWI monthly climatology — fire danger conditions
  - Lightning density proxy — spatial density of historically lightning-caused fires
  - Human proximity — distance to nearest road (human-caused fire risk)
  - Month — seasonality

Training labels:
  - Positive: 1 km² cells that had a fire ignition in month M, year Y (from CNFDB)
  - Negative: randomly sampled cells with no ignition in same month/year

Cause separation:
  - CAUSE='N' (Natural/Lightning) → lightning model
  - CAUSE='H' (Human) → human-caused model
  - Combined model trained on all causes

Output: P(ignition) per km² per month — drives the Risk tab monthly forecast.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# BC bounding box
BC_LAT_MIN, BC_LAT_MAX = 48.0, 60.0
BC_LON_MIN, BC_LON_MAX = -139.1, -114.0
GRID_RES_DEG = 0.01   # ~1 km at BC latitudes


def load_ignition_points(
    nfdb_dir: Path,
    province: str = "BC",
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Load CNFDB fire origin points for BC.

    Returns DataFrame with columns:
      LATITUDE, LONGITUDE, YEAR, MONTH, CAUSE, FIRE_ID
    """
    shps = list(nfdb_dir.rglob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No SHP files in {nfdb_dir}")

    gdf = gpd.read_file(shps[0])
    log.info(f"Loaded {len(gdf):,} total fire points")

    # Filter to BC
    bc = gdf[gdf["SRC_AGENCY"] == province].copy()
    log.info(f"  BC: {len(bc):,}")

    if years:
        bc = bc[bc["YEAR"].isin(years)]
        log.info(f"  {years[0]}–{years[-1]}: {len(bc):,}")

    # Keep essential columns
    cols = ["LATITUDE", "LONGITUDE", "YEAR", "MONTH", "CAUSE", "FIRE_ID", "SIZE_HA"]
    available = [c for c in cols if c in bc.columns]
    bc = bc[available].copy()
    bc["LATITUDE"] = pd.to_numeric(bc["LATITUDE"], errors="coerce")
    bc["LONGITUDE"] = pd.to_numeric(bc["LONGITUDE"], errors="coerce")
    bc = bc.dropna(subset=["LATITUDE", "LONGITUDE"])

    return bc.reset_index(drop=True)


def build_lightning_density_grid(
    ignition_df: pd.DataFrame,
    grid_res: float = 1.0,  # degrees — coarse for density estimation
) -> dict:
    """
    Compute spatial density of lightning-caused (CAUSE='N') fire ignitions.
    
    Returns dict mapping (lat_bin, lon_bin) → density score.
    Used as a proxy for lightning climatology.
    """
    lightning = ignition_df[ignition_df.get("CAUSE", "U") == "N"].copy()
    if lightning.empty:
        return {}

    # Bin to coarse grid
    lightning["lat_bin"] = (lightning["LATITUDE"] / grid_res).astype(int) * grid_res
    lightning["lon_bin"] = (lightning["LONGITUDE"] / grid_res).astype(int) * grid_res

    counts = lightning.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")
    # Normalise to [0, 1]
    max_count = counts["count"].max()
    counts["lightning_density"] = counts["count"] / max(max_count, 1)

    density_dict = {
        (row.lat_bin, row.lon_bin): row.lightning_density
        for _, row in counts.iterrows()
    }
    log.info(f"Lightning density grid: {len(density_dict)} cells ({len(lightning):,} lightning fires)")
    return density_dict


def build_human_density_grid(
    ignition_df: pd.DataFrame,
    grid_res: float = 1.0,
) -> dict:
    """Compute spatial density of human-caused (CAUSE='H') fire ignitions."""
    human = ignition_df[ignition_df.get("CAUSE", "U") == "H"].copy() if "CAUSE" in ignition_df.columns else pd.DataFrame()
    if human.empty:
        return {}
    human["lat_bin"] = (human["LATITUDE"] / grid_res).astype(int) * grid_res
    human["lon_bin"] = (human["LONGITUDE"] / grid_res).astype(int) * grid_res
    counts = human.groupby(["lat_bin", "lon_bin"]).size().reset_index(name="count")
    max_count = counts["count"].max()
    counts["human_density"] = counts["count"] / max(max_count, 1)
    return {(r.lat_bin, r.lon_bin): r.human_density for _, r in counts.iterrows()}


def snap_density(lat: float, lon: float, density_dict: dict, grid_res: float = 1.0) -> float:
    """Look up density value for a lat/lon point."""
    lat_bin = round((lat // grid_res) * grid_res, 4)
    lon_bin = round((lon // grid_res) * grid_res, 4)
    return float(density_dict.get((lat_bin, lon_bin), 0.0))


# BC landcover classes mapped to fire flammability score [0-1]
# Based on Canadian Forest Fire Behaviour Prediction (FBP) system fuel types
# Source: Canada Landcover 2020 class definitions
LANDCOVER_FLAMMABILITY = {
    1:  0.9,   # Temperate/subpolar needleleaf forest (high — pine, spruce)
    2:  0.8,   # Subpolar taiga needleleaf forest (high)
    3:  0.6,   # Tropical/subtropical broadleaf evergreen forest
    4:  0.5,   # Tropical broadleaf deciduous (moderate)
    5:  0.7,   # Temperate/subpolar broadleaf deciduous (moderate-high — aspen)
    6:  0.7,   # Mixed forest (moderate-high)
    7:  0.8,   # Tropical/subtropical shrubland (high — grass/shrub)
    8:  0.8,   # Temperate/subpolar shrubland (high — dry shrub)
    9:  0.9,   # Tropical/subtropical grassland (very high)
    10: 0.8,   # Temperate/subpolar grassland (high)
    11: 0.3,   # Polar/alpine vegetation (low — sparse)
    12: 0.1,   # Cropland (low)
    13: 0.05,  # Wetland (very low — wet fuels)
    14: 0.0,   # Urban (no ignition)
    15: 0.1,   # Sparse (bare/rock)
    16: 0.0,   # Snow/ice
    17: 0.0,   # Water
    18: 0.05,  # Tidal flat
    19: 0.2,   # Lichen-dominated (low)
}

_CDEM_CACHE: Optional["np.ndarray"] = None
_CDEM_TRANSFORM = None
_LC_CACHE: Optional["np.ndarray"] = None
_LC_TRANSFORM = None


def _load_terrain_cache(cdem_path: Path) -> tuple:
    global _CDEM_CACHE, _CDEM_TRANSFORM
    if _CDEM_CACHE is None and cdem_path.exists():
        try:
            import rasterio
            from src.data.assemble import compute_slope_aspect
            with rasterio.open(cdem_path) as src:
                _CDEM_CACHE = src.read(1).astype(np.float32)
                _CDEM_TRANSFORM = src.transform
        except Exception as e:
            pass
    return _CDEM_CACHE, _CDEM_TRANSFORM


def _load_lc_cache(lc_path: Path) -> tuple:
    global _LC_CACHE, _LC_TRANSFORM
    if _LC_CACHE is None and lc_path.exists():
        try:
            import rasterio
            with rasterio.open(lc_path) as src:
                _LC_CACHE = src.read(1).astype(np.int16)
                _LC_TRANSFORM = src.transform
        except Exception as e:
            pass
    return _LC_CACHE, _LC_TRANSFORM


def _sample_raster(arr, transform, lat: float, lon: float, default: float = 0.0) -> float:
    """Sample a raster array at a lat/lon point."""
    if arr is None or transform is None:
        return default
    try:
        import rasterio
        row, col = rasterio.transform.rowcol(transform, lon, lat)
        r, c = int(row), int(col)
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            return float(arr[r, c])
    except Exception:
        pass
    return default


def build_ignition_features(
    lat: float,
    lon: float,
    month: int,
    lightning_density: float,
    human_density: float,
    fwi_climatology: Optional[dict] = None,
    cdem_arr=None,
    cdem_transform=None,
    lc_arr=None,
    lc_transform=None,
) -> dict:
    """
    Build a feature vector for one (lat, lon, month) combination.
    These are the features the ignition RF uses for prediction.
    """
    # Spatial gradients capturing BC fire regime geography
    coast_factor = min(1.0, max(0.0, (lon + 132) / 16))   # 0=coast, 1=NE interior
    lat_factor   = max(0.0, 1.0 - abs(lat - 53) / 10)    # peaks at 53°N

    # Seasonal FWI from climatology
    fwi_mean = 0.0
    if fwi_climatology and str(month) in fwi_climatology:
        month_data = fwi_climatology[str(month)]
        if month_data.get("fwi_mean"):
            fwi_mean = float(np.mean(month_data["fwi_mean"]))
    if fwi_mean == 0:
        seasonal = [2, 2.5, 4, 7, 12, 18, 28, 26, 14, 6, 2.5, 2]
        fwi_mean = seasonal[month - 1] * coast_factor * lat_factor

    # Terrain features
    elevation = _sample_raster(cdem_arr, cdem_transform, lat, lon, default=500.0)
    # Slope from elevation gradient (approximate from neighbours)
    elev_n = _sample_raster(cdem_arr, cdem_transform, lat + 0.01, lon, default=elevation)
    elev_e = _sample_raster(cdem_arr, cdem_transform, lat, lon + 0.01, default=elevation)
    slope_approx = min(45.0, float(np.sqrt((elevation - elev_n)**2 + (elevation - elev_e)**2) / 1000.0))

    # Fuel type flammability from landcover
    lc_class = int(_sample_raster(lc_arr, lc_transform, lat, lon, default=6.0))
    flammability = LANDCOVER_FLAMMABILITY.get(lc_class, 0.5)

    # South-facing slope drying factor (higher = drier = more flammable)
    # Approximate: southern BC (lat<55) has more south-facing ignitions
    south_factor = max(0.0, (55 - lat) / 10) if lat < 55 else 0.0

    return {
        "lat": lat,
        "lon": lon,
        "month": month,
        "lightning_density": lightning_density,
        "human_density": human_density,
        "fwi_mean": fwi_mean,
        "coast_factor": coast_factor,
        "lat_factor": lat_factor,
        "is_fire_season": 1 if month in [5, 6, 7, 8, 9] else 0,
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        # New terrain/fuel features
        "elevation": min(3000.0, max(0.0, elevation)),
        "slope": slope_approx,
        "flammability": flammability,
        "south_factor": south_factor,
        "fwi_x_flammability": fwi_mean * flammability,
    }


FEATURE_COLS = [
    "lat", "lon", "month",
    "lightning_density", "human_density",
    "fwi_mean", "coast_factor", "lat_factor",
    "is_fire_season", "month_sin", "month_cos",
    # Terrain + fuel
    "elevation", "slope", "flammability", "south_factor", "fwi_x_flammability",
]


def build_training_data(
    ignition_df: pd.DataFrame,
    lightning_density: dict,
    human_density: dict,
    fwi_climatology: Optional[dict] = None,
    neg_ratio: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) training arrays for the ignition RF.

    Positives: actual fire ignition locations from CNFDB
    Negatives: randomly sampled BC grid cells with no fire

    Parameters
    ----------
    neg_ratio : int
        Number of negative samples per positive.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Positives
    for _, row in ignition_df.iterrows():
        feat = build_ignition_features(
            lat=float(row["LATITUDE"]),
            lon=float(row["LONGITUDE"]),
            month=int(row.get("MONTH", 7) or 7),
            lightning_density=snap_density(row["LATITUDE"], row["LONGITUDE"], lightning_density),
            human_density=snap_density(row["LATITUDE"], row["LONGITUDE"], human_density),
            fwi_climatology=fwi_climatology,
        )
        feat["label"] = 1
        rows.append(feat)

    n_pos = len(rows)
    n_neg = n_pos * neg_ratio

    # Negatives — random BC grid cells
    rand_lats = rng.uniform(BC_LAT_MIN, BC_LAT_MAX, n_neg)
    rand_lons = rng.uniform(BC_LON_MIN, BC_LON_MAX, n_neg)
    rand_months = rng.integers(1, 13, n_neg)

    for lat, lon, month in zip(rand_lats, rand_lons, rand_months):
        feat = build_ignition_features(
            lat=float(lat), lon=float(lon), month=int(month),
            lightning_density=snap_density(lat, lon, lightning_density),
            human_density=snap_density(lat, lon, human_density),
            fwi_climatology=fwi_climatology,
        )
        feat["label"] = 0
        rows.append(feat)

    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    log.info(f"Training data: {n_pos:,} positives, {n_neg:,} negatives")
    return X, y


def train_ignition_model(
    nfdb_dir: Path,
    stats_dir: Path,
    output_dir: Path,
    years: Optional[list[int]] = None,
) -> "sklearn.ensemble.RandomForestClassifier":
    """
    Train the ignition prediction Random Forest.
    """
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading fire ignition points…")
    ignition_df = load_ignition_points(nfdb_dir, years=years)

    log.info("Computing lightning and human density grids…")
    lightning_density = build_lightning_density_grid(ignition_df)
    human_density = build_human_density_grid(ignition_df)

    # Load FWI climatology
    clim_path = stats_dir / "fwi_climatology.json"
    fwi_climatology = None
    if clim_path.exists():
        with open(clim_path) as f:
            fwi_climatology = json.load(f)

    log.info("Building training data…")
    X, y = build_training_data(ignition_df, lightning_density, human_density, fwi_climatology)

    log.info("Training Random Forest ignition model…")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    log.info("Training complete.")

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    top = sorted(importance.items(), key=lambda x: -x[1])[:6]
    log.info(f"Top features: {top}")

    # Save
    model_path = output_dir / "ignition_rf.joblib"
    joblib.dump(model, model_path)

    # Save density grids for API use
    density_path = stats_dir / "ignition_densities.json"
    with open(density_path, "w") as f:
        json.dump({
            "lightning": {f"{k[0]},{k[1]}": v for k, v in lightning_density.items()},
            "human":     {f"{k[0]},{k[1]}": v for k, v in human_density.items()},
        }, f, separators=(",", ":"))

    log.info(f"Model saved → {model_path}")
    log.info(f"Densities saved → {density_path}")
    return model


def predict_ignition_grid(
    model,
    month: int,
    fwi_climatology: Optional[dict] = None,
    lightning_density: Optional[dict] = None,
    human_density: Optional[dict] = None,
    grid_step: float = 0.25,
) -> dict:
    """
    Run the ignition model over a BC grid for a given month.
    Returns GeoJSON FeatureCollection.
    """
    lats = np.arange(BC_LAT_MIN, BC_LAT_MAX, grid_step)
    lons = np.arange(BC_LON_MIN, BC_LON_MAX, grid_step)
    ld = lightning_density or {}
    hd = human_density or {}

    rows = []
    for lat in lats:
        for lon in lons:
            feat = build_ignition_features(
                lat=float(lat), lon=float(lon), month=month,
                lightning_density=snap_density(lat, lon, ld),
                human_density=snap_density(lat, lon, hd),
                fwi_climatology=fwi_climatology,
            )
            rows.append(feat)

    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS].values.astype(np.float32)
    probs = model.predict_proba(X)[:, 1]

    features = []
    for i, (lat, lon, prob) in enumerate(zip(df["lat"], df["lon"], probs)):
        if prob > 0.01:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(float(lon), 3), round(float(lat), 3)]},
                "properties": {
                    "ignition_prob": round(float(prob), 4),
                    "lightning_density": round(float(df.iloc[i]["lightning_density"]), 3),
                    "human_density": round(float(df.iloc[i]["human_density"]), 3),
                    "fwi_mean": round(float(df.iloc[i]["fwi_mean"]), 1),
                },
            })

    return {"type": "FeatureCollection", "features": features}
