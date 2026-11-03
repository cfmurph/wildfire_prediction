#!/usr/bin/env python3
"""
Live Spread Prediction
=======================
Where each actively burning BC fire is most likely to grow next, as GeoJSON for the map.

For a day T (default yesterday, so its night satellite pass is in), each fire BCWS lists as
not out gets the same 64 km patches as in training. The burn mask comes from VIIRS detections
since ignition, kept where they connect to a BCWS perimeter or fire location; patches with no
detection on T or the day before are skipped as not actively burning. The U-Net checkpoints
are averaged and cut into two bands, set on the 2021–22 validation seasons:

  highest   the top 1% of unburned land near fires (in testing, 40% of next-day growth)
  elevated  the top 5% (82%)

Bands stop 15 km from the burn, the reach of 97% of next-day growth in training: on extreme
fire-weather days the model otherwise rates land far beyond any one day's run.

Usage:
  python -m scripts.predict_spread [--date 2026-08-14]   (from the repo root)

Needs FIRMS_MAP_KEY (api/.env) and the raw static data from scripts/download_canada.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape

from api.services import bc_fires
from src.data.assemble import OUTPUT_CRS, PATCH_EXTENT_M, fire_tiles, firms_daily_masks
from src.data.dataset import ChannelStats
from src.data.live import LiveAssembler, firms_detections
from src.models.unet import build_unet
from src.utils.config import N_CHANNELS, PATCH_SIZE, TARGET_RESOLUTION_M

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Ensemble probability cut-offs: top 1% / 5% of unburned patch pixels, 2021–22 validation
BANDS = [("highest", 0.944), ("elevated", 0.539)]
MAX_REACH_KM = 15       # 97% of next-day new burn in 2012–22 landed within this of the day-T burn
MAX_HISTORY_DAYS = 120  # burn before this counts as nothing; long enough for any active fire
MODEL_DIR = Path("experiments/checkpoints/spread")  # the checkpoints and the stats they were trained with


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict next-day growth of active BC fires")
    p.add_argument("--checkpoint", nargs="+", type=Path, default=sorted(MODEL_DIR.glob("*.pt")),
                   help="U-Net checkpoints to average")
    p.add_argument("--date", help="day T, YYYY-MM-DD (default: yesterday, UTC)")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--stats", type=Path, default=MODEL_DIR / "channel_stats.json")
    p.add_argument("--cache-dir", type=Path, default=Path("data/live/cache"))
    p.add_argument("--out", type=Path, default=Path("data/live/spread.geojson"))
    return p.parse_args()


def known_fires(year: int) -> gpd.GeoDataFrame:
    """BCWS fires not yet out, with their perimeter where one is mapped (else the point), in the patch CRS."""
    points = asyncio.run(bc_fires.get_active_fires())
    if points is None:  # an outage must not read as a day with no fires
        raise RuntimeError("Could not fetch the BCWS active fires")
    perims = asyncio.run(bc_fires.get_historical_perimeters(year)) or {"features": []}
    by_number = {f["properties"].get("fire_number"): shape(f["geometry"]) for f in perims["features"] if f["geometry"]}
    rows = [
        {"fire_number": f["properties"]["fire_number"], "fire_name": f["properties"].get("fire_name", ""),
         "ignition_date": f["properties"].get("ignition_date"),
         "geometry": by_number.get(f["properties"]["fire_number"]) or shape(f["geometry"])}
        for f in points["features"]
    ]
    columns = ["fire_number", "fire_name", "ignition_date", "geometry"]
    return gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs="EPSG:4326").to_crs(OUTPUT_CRS)


def spread_bands(prob: np.ndarray, burned: np.ndarray) -> np.ndarray:
    """Band level per pixel (1 is BANDS[0], 0 is none): unburned land within reach of the burn."""
    band = np.zeros(prob.shape, dtype=np.uint8)
    for level, (_, cut) in reversed(list(enumerate(BANDS, start=1))):
        band[prob >= cut] = level
    band[burned > 0] = 0
    band[distance_transform_edt(burned == 0) * TARGET_RESOLUTION_M / 1000 > MAX_REACH_KM] = 0
    return band


def main() -> None:
    args = parse_args()
    # ChannelStats falls back to no normalisation, and the models would then flag nothing without complaint
    if not (args.checkpoint and args.stats.exists()):
        raise FileNotFoundError(f"Need checkpoints and stats, got {args.checkpoint} and {args.stats}")
    load_dotenv("api/.env")
    yesterday = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    day = pd.Timestamp(args.date) if args.date else yesterday
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    fires = known_fires(day.year)
    log.info(f"{len(fires)} BCWS fires not out")
    start = day - pd.Timedelta(days=MAX_HISTORY_DAYS)
    detections = firms_detections(start, day, os.environ["FIRMS_MAP_KEY"], args.cache_dir)
    dates = list(pd.date_range(start, day))

    cfg = OmegaConf.load("configs/unet.yaml").model
    models = []
    for path in args.checkpoint:
        m = build_unet(N_CHANNELS, cfg.base_filters, cfg.depth, cfg.dropout)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models.append(m.eval())
    stats = ChannelStats(args.stats)
    assembler = LiveAssembler(args.raw_dir, args.cache_dir)
    assembler.fire_history = assembler._load_fire_history(assembler._load_perimeters())

    shape_ = (PATCH_SIZE, PATCH_SIZE)
    # Any known fire counts, as training counted any of the season's perimeters
    buffered = fires.geometry.buffer(1000)
    features, done = [], set()
    for fire in fires.itertuples():
        for xmin, ymax in fire_tiles(*buffered[fire.Index].bounds):
            if (xmin, ymax) in done:
                continue
            done.add((xmin, ymax))
            transform = from_bounds(xmin, ymax - PATCH_EXTENT_M, xmin + PATCH_EXTENT_M, ymax, *shape_)
            known = rasterio.features.rasterize(
                [(g, 1.0) for g in buffered.cx[xmin:xmin + PATCH_EXTENT_M, ymax - PATCH_EXTENT_M:ymax]],
                out_shape=shape_, transform=transform, dtype=np.float32, fill=0.0,
            )
            masks = firms_daily_masks(detections, dates, transform, known)
            if not (masks[-1] > masks[-3]).any():
                continue  # no new detections on T or T-1: not actively burning here
            X = assembler._build_patch_channels(f"{day:%Y-%m-%d}", masks[-1], transform, shape_, day.year)
            x = torch.from_numpy(stats.normalize(X).astype(np.float32))[None]
            with torch.no_grad():
                prob = np.mean([torch.sigmoid(m(x))[0, 0].numpy() for m in models], axis=0)
            band = spread_bands(prob, masks[-1])
            for geom, level in rasterio.features.shapes(band, mask=band > 0, transform=transform):
                features.append({"type": "Feature", "geometry": geom,
                                 "properties": {"band": BANDS[int(level) - 1][0], "fire_number": fire.fire_number}})
            log.info(f"  {fire.fire_number} {fire.fire_name}: {int((band > 0).sum())} km² flagged")

    out = gpd.GeoDataFrame.from_features(features, crs=OUTPUT_CRS).to_crs("EPSG:4326") if features else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    collection = json.loads(out.to_json()) if out is not None else {"type": "FeatureCollection", "features": []}
    collection.update(date=f"{day:%Y-%m-%d}", issued=datetime.now(timezone.utc).isoformat(timespec="minutes"))
    args.out.write_text(json.dumps(collection))
    log.info(f"Wrote {len(collection['features'])} areas for {day:%Y-%m-%d} to {args.out}")


if __name__ == "__main__":
    main()
