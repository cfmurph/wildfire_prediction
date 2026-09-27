"""
Fire spread prediction service.

Phase 1: Physics-inspired heuristic spread model.
         Returns a GeoJSON polygon representing predicted next-day perimeter.
         Slots in for the ConvLSTM/U-Net model once trained.

The heuristic:
  - Centre on the fire's current location
  - Apply a wind-biased anisotropic Gaussian kernel
  - Scale spread radius by FWI and wind speed
  - Return 3 probability contours: 25th / 50th / 75th percentile

This is NOT the ML model — it's a physics proxy so the frontend shows
meaningful spatial predictions before training completes.

Phase 2: load trained model artifact from MLflow and run inference.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

log = logging.getLogger(__name__)

# Base spread rate (km / h) under FWI=1, wind=1 m/s — calibrated to BC interior
BASE_SPREAD_KM_PER_H = 0.02

# Anisotropy: ratio of cross-wind to along-wind spread
CROSSWIND_RATIO = 0.35


def spread_forecast(
    lat: float,
    lon: float,
    wind_dir_deg: float = 270.0,   # meteorological: direction FROM which wind blows
    wind_speed_ms: float = 5.0,
    fwi: float = 20.0,
    isi: float = 8.0,
    hours: float = 24.0,
    n_points: int = 64,
) -> dict:
    """
    Generate a GeoJSON spread forecast for a fire at (lat, lon).

    Returns a FeatureCollection with three polygon features:
      - p50: median spread (most likely perimeter)
      - p75: 75th percentile (upper bound)
      - p25: 25th percentile (conservative)

    Parameters
    ----------
    lat, lon : fire centroid
    wind_dir_deg : meteorological wind direction (FROM which it blows)
                   0° = north, 90° = east, 180° = south, 270° = west
    wind_speed_ms : wind speed in m/s
    fwi : Fire Weather Index
    isi : Initial Spread Index (more directly linked to ROS)
    hours : prediction horizon in hours
    n_points : polygon resolution
    """
    # Spread direction = wind is FROM wind_dir, fire spreads INTO wind
    # Meteorological convention: 270° = westerly wind → fire spreads east
    spread_dir_deg = (wind_dir_deg + 180) % 360

    # Radial spread distance (km) — ISI is more directly linked to ROS than FWI
    # ISI scale: 0-5 low, 5-15 moderate, 15-30 high, >30 extreme
    fwi_factor = math.log1p(max(fwi, 1)) / math.log1p(20)   # normalised around FWI=20
    wind_factor = math.log1p(wind_speed_ms) / math.log1p(8)  # normalised around 8 m/s
    isi_factor = math.log1p(max(isi, 1)) / math.log1p(10)

    # Major axis (along-wind) spread radius in km
    r_major_p50 = BASE_SPREAD_KM_PER_H * hours * fwi_factor * wind_factor * (1 + isi_factor)
    r_major_p50 = max(r_major_p50, 0.5)   # minimum 0.5 km

    # Uncertainty: 25th / 75th percentile as fractions of p50
    r_major_p25 = r_major_p50 * 0.5
    r_major_p75 = r_major_p50 * 1.8

    # Minor axis (cross-wind)
    r_minor_p50 = r_major_p50 * CROSSWIND_RATIO
    r_minor_p25 = r_major_p25 * CROSSWIND_RATIO
    r_minor_p75 = r_major_p75 * CROSSWIND_RATIO

    features = []
    for label, r_maj, r_min, prob in [
        ("p25", r_major_p25, r_minor_p25, 0.25),
        ("p50", r_major_p50, r_minor_p50, 0.50),
        ("p75", r_major_p75, r_minor_p75, 0.75),
    ]:
        polygon_coords = _ellipse_polygon(
            lat=lat,
            lon=lon,
            r_major_km=r_maj,
            r_minor_km=r_min,
            rotation_deg=spread_dir_deg,
            n_points=n_points,
        )
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords],
            },
            "properties": {
                "percentile": label,
                "burn_probability": prob,
                "radius_major_km": round(r_maj, 2),
                "radius_minor_km": round(r_min, 2),
                "spread_dir_deg": spread_dir_deg,
                "wind_dir_deg": wind_dir_deg,
                "wind_speed_ms": wind_speed_ms,
                "fwi": fwi,
                "isi": isi,
                "hours": hours,
                "model": "heuristic_v1",   # will become "convlstm_v1" after training
                "area_ha": round(math.pi * r_maj * r_min * 100, 0),  # km² → ha
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "model": "heuristic_v1",
            "note": "Physics-inspired heuristic. ML model (ConvLSTM) replaces this after training.",
            "lat": lat,
            "lon": lon,
            "hours": hours,
        },
    }


def _ellipse_polygon(
    lat: float,
    lon: float,
    r_major_km: float,
    r_minor_km: float,
    rotation_deg: float,
    n_points: int = 64,
) -> list[list[float]]:
    """
    Generate a GeoJSON polygon (list of [lon, lat] pairs) for an ellipse.

    The ellipse is centred at (lat, lon), with:
      - r_major_km along the rotation_deg direction
      - r_minor_km perpendicular to it
    """
    # Degrees per km at BC latitudes (~53°N)
    lat_deg_per_km = 1 / 111.0
    lon_deg_per_km = 1 / (111.0 * math.cos(math.radians(lat)))

    rot_rad = math.radians(rotation_deg)
    coords = []

    for i in range(n_points + 1):
        theta = 2 * math.pi * i / n_points
        # Ellipse in local frame (major axis along rotation)
        x_local = r_major_km * math.cos(theta)
        y_local = r_minor_km * math.sin(theta)

        # Rotate by spread direction (clockwise from north = azimuth)
        x_rot = x_local * math.sin(rot_rad) + y_local * math.cos(rot_rad)
        y_rot = x_local * math.cos(rot_rad) - y_local * math.sin(rot_rad)

        coords.append([
            round(lon + x_rot * lon_deg_per_km, 6),
            round(lat + y_rot * lat_deg_per_km, 6),
        ])

    return coords


def risk_grid(n_lat: int = 40, n_lon: int = 40) -> dict:
    """
    Generate a long-term burn likelihood grid for BC.

    Phase 1: climatology-based estimate from historical burn frequency patterns.
    Uses BC Interior fire climatology — highest risk in central/NE interior.

    Phase 2: replace with BurnP3+ outputs or trained risk model.
    """
    lat_min, lat_max = 49.0, 59.5
    lon_min, lon_max = -139.0, -114.5

    features = []
    for i in range(n_lat):
        for j in range(n_lon):
            lat = lat_min + (lat_max - lat_min) * i / (n_lat - 1)
            lon = lon_min + (lon_max - lon_min) * j / (n_lon - 1)

            # Risk model: peaks in BC Interior (roughly 52-56°N, 120-130°W)
            lat_peak = math.exp(-((lat - 53) ** 2) / 18)
            lon_peak = math.exp(-((lon + 124) ** 2) / 30)
            coast_penalty = max(0, 1 - (lon + 130) / 5)   # suppress near coast

            risk = min(1.0, lat_peak * lon_peak * 0.85 * (1 - coast_penalty * 0.4))

            if risk < 0.02:
                continue   # skip very low risk cells to reduce payload

            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 3), round(lat, 3)]},
                "properties": {
                    "burn_probability": round(risk, 3),
                    "risk_class": _risk_class(risk),
                },
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "model": "climatology_v1",
            "note": "Historical burn frequency climatology. BurnP3+ replaces this in Phase 2.",
            "resolution_deg": round((lat_max - lat_min) / n_lat, 3),
        },
    }


def _risk_class(p: float) -> str:
    if p < 0.1:  return "Low"
    if p < 0.3:  return "Moderate"
    if p < 0.6:  return "High"
    return "Very High"
