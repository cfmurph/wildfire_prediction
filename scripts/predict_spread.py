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

Needs FIRMS_MAP_KEY and DATABASE_URL (api/.env) and the raw static data from scripts/download_canada.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
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

from api.services import bc_fires, spread_store
from scripts.backup import backup
from src.data.assemble import OUTPUT_CRS, PATCH_EXTENT_M, fire_tiles, firms_daily_masks
from src.data.dataset import ChannelStats
from src.data.live import LiveAssembler, archive_inputs, firms_detections, prune_cache
from src.models.unet import build_unet
from src.utils.config import N_CHANNELS, PATCH_SIZE, TARGET_RESOLUTION_M

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Ensemble probability cut-offs: top 1% / 5% of unburned patch pixels, 2021–22 validation
BANDS = [("highest", 0.944), ("elevated", 0.539)]
MAX_REACH_KM = 15       # 97% of next-day new burn in 2012–22 landed within this of the day-T burn
MAX_HISTORY_DAYS = 120  # burn before this counts as nothing; long enough for any active fire
ARCHIVE_WITHIN_DAYS = 4  # ECMWF's forecasts are gone after this, so a rerun of an older day is not worth keeping
MAX_SCORE_LAG_DAYS = 7  # a day still unscored after this is left, rather than reported every morning
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
    p.add_argument("--archive-dir", type=Path, default=Path("data/live/archive"))
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


def predict_fires(fires, buffered, predict_tile) -> tuple[list[dict], list[str]]:
    """
    predict_tile's features over every fire's tiles, and the fires a tile failed for. predict_tile
    returns None for a tile not burning. A failed tile costs only itself, unless none was predicted
    at all: that is a broken run, such as a missing day of weather, not a quiet day.
    """
    features, failed, done, predicted = [], [], set(), 0
    for fire in fires.itertuples():
        for xmin, ymax in fire_tiles(*buffered[fire.Index].bounds):
            if (xmin, ymax) in done:
                continue
            done.add((xmin, ymax))
            try:
                tile_features = predict_tile(fire, xmin, ymax)
            except Exception:
                log.exception(f"Could not predict a tile of {fire.fire_number}")
                failed.append(fire.fire_number)
                continue
            if tile_features is not None:
                predicted += 1
                features.extend(tile_features)
    if failed and not predicted:
        raise RuntimeError(f"Every burning tile failed: {sorted(set(failed))}")
    return features, sorted(set(failed))


def tile_masks(xmin: float, ymax: float, buffered, detections, dates) -> tuple:
    """A tile's transform and its daily burn masks, counting detections that connect to a known fire."""
    transform = from_bounds(xmin, ymax - PATCH_EXTENT_M, xmin + PATCH_EXTENT_M, ymax, PATCH_SIZE, PATCH_SIZE)
    known = rasterio.features.rasterize(
        [(g, 1.0) for g in buffered.cx[xmin:xmin + PATCH_EXTENT_M, ymax - PATCH_EXTENT_M:ymax]],
        out_shape=(PATCH_SIZE, PATCH_SIZE), transform=transform, dtype=np.float32, fill=0.0,
    )
    return transform, firms_daily_masks(detections, dates, transform, known)


def tile_score(fire_number: str, band: np.ndarray, burned: np.ndarray, new_burn: np.ndarray) -> dict:
    """How a tile's bands (1 highest, 2 elevated) did against the next day's new burn, in km² (1 km pixels)."""
    return {
        "fire_number": fire_number,
        "unburned_km2": int((burned == 0).sum()),
        "new_burn_km2": int(new_burn.sum()),
        "flagged_highest_km2": int((band == 1).sum()),
        "flagged_elevated_km2": int((band == 2).sum()),
        "hit_highest_km2": int(((band == 1) & new_burn).sum()),
        "hit_elevated_km2": int(((band == 2) & new_burn).sum()),
    }


def score_run(database_url: str, date_str: str, detections, dates: list, archive_dir: Path) -> list[str]:
    """
    Score a day's prediction against the next day's detections, by the fires listed that day, so
    tiles match the prediction's even if a fire has since left the list. Returns fires that failed.
    """
    fires = gpd.read_file(archive_dir / date_str / "fires.geojson").to_crs(OUTPUT_CRS)
    buffered = fires.geometry.buffer(1000)
    features = spread_store.run_features(database_url, date_str)
    areas = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326").to_crs(OUTPUT_CRS) if features else None
    through_next_day = dates[:dates.index(pd.Timestamp(date_str) + pd.Timedelta(days=1)) + 1]

    def score_tile(fire, xmin: float, ymax: float) -> list[dict] | None:
        transform, masks = tile_masks(xmin, ymax, buffered, detections, through_next_day)
        new_burn = (masks[-1] > 0) & (masks[-2] == 0)
        band = np.zeros(new_burn.shape, dtype=np.uint8)
        if areas is not None:
            in_tile = areas.cx[xmin:xmin + PATCH_EXTENT_M, ymax - PATCH_EXTENT_M:ymax]
            shapes = [(g, level) for level, (name, _) in enumerate(BANDS, start=1)
                      for g in in_tile.geometry[in_tile["band"] == name]]
            if shapes:
                band = rasterio.features.rasterize(shapes, out_shape=band.shape, transform=transform, dtype=np.uint8)
        if not (band.any() or new_burn.any()):
            return None
        return [tile_score(fire.fire_number, band, masks[-2], new_burn)]

    rows, failed = predict_fires(fires, buffered, score_tile)
    # A fire wider than one tile has several rows
    scores = pd.DataFrame(rows).groupby("fire_number", as_index=False).sum().to_dict("records") if rows else []
    spread_store.save_scores(database_url, date_str, scores)
    log.info(f"Scored {date_str}: {sum(s['new_burn_km2'] for s in scores)} km² of new burn, "
             f"{sum(s['hit_highest_km2'] + s['hit_elevated_km2'] for s in scores)} km² of it flagged")
    return failed


def notify(message: str) -> None:
    """A macOS notification, since nobody reads the job's log until something looks wrong."""
    if sys.platform == "darwin":
        script = f'display notification {json.dumps(message)} with title "Wildfire spread run"'
        subprocess.run(["osascript", "-e", script], check=False)


def main() -> None:
    args = parse_args()
    try:
        problems = run(args)
    except Exception as exc:
        notify(f"Failed with {type(exc).__name__}, so the map was not updated. See data/live/spread.log")
        raise
    if problems:
        notify(f"Finished with problems: {'; '.join(problems)}. See data/live/spread.log")


def run(args: argparse.Namespace) -> list[str]:
    """Predict and archive one day; returns what went wrong without stopping the run."""
    problems = []
    # ChannelStats falls back to no normalisation, and the models would then flag nothing without complaint
    if not (args.checkpoint and args.stats.exists()):
        raise FileNotFoundError(f"Need checkpoints and stats, got {args.checkpoint} and {args.stats}")
    load_dotenv("api/.env")
    yesterday = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=1)
    day = pd.Timestamp(args.date) if args.date else yesterday
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    fires = known_fires(day.year)
    log.info(f"{len(fires)} BCWS fires not out")
    # First, so a failure later in the run does not lose the day's inputs
    if (yesterday - day).days <= ARCHIVE_WITHIN_DAYS:
        saved, not_archived = archive_inputs(day, fires, os.environ["FIRMS_MAP_KEY"], args.cache_dir, args.archive_dir)
        if not_archived:
            problems.append(f"not archived: {', '.join(not_archived)}")
        log.info(f"Archived {len(saved)} input files for {day:%Y-%m-%d}")
        if database_url := os.environ.get("DATABASE_URL"):
            spread_store.save_archive(database_url, f"{day:%Y-%m-%d}", saved)
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

    def predict_tile(fire, xmin: float, ymax: float) -> list[dict] | None:
        transform, masks = tile_masks(xmin, ymax, buffered, detections, dates)
        if not (masks[-1] > masks[-3]).any():
            return None  # no new detections on T or T-1: not actively burning here
        X = assembler._build_patch_channels(f"{day:%Y-%m-%d}", masks[-1], transform, shape_, day.year)
        x = torch.from_numpy(stats.normalize(X).astype(np.float32))[None]
        with torch.no_grad():
            prob = np.mean([torch.sigmoid(m(x))[0, 0].numpy() for m in models], axis=0)
        band = spread_bands(prob, masks[-1])
        log.info(f"  {fire.fire_number} {fire.fire_name}: {int((band > 0).sum())} km² flagged")
        return [{"type": "Feature", "geometry": geom,
                 "properties": {"band": BANDS[int(level) - 1][0], "fire_number": fire.fire_number}}
                for geom, level in rasterio.features.shapes(band, mask=band > 0, transform=transform)]

    features, failed_fires = predict_fires(fires, buffered, predict_tile)
    if failed_fires:
        problems.append(f"no prediction for {', '.join(failed_fires)}")

    out = gpd.GeoDataFrame.from_features(features, crs=OUTPUT_CRS).to_crs("EPSG:4326") if features else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    collection = json.loads(out.to_json()) if out is not None else {"type": "FeatureCollection", "features": []}
    collection.update(date=f"{day:%Y-%m-%d}", issued=datetime.now(timezone.utc).isoformat(timespec="minutes"),
                      failed_fires=failed_fires)
    args.out.write_text(json.dumps(collection))
    prune_cache(args.cache_dir)
    if database_url := os.environ.get("DATABASE_URL"):
        spread_store.save_run(database_url, collection)
        since = f"{day - pd.Timedelta(days=MAX_SCORE_LAG_DAYS):%Y-%m-%d}"
        for date_str in spread_store.unscored_runs(database_url, since, f"{day:%Y-%m-%d}"):
            try:
                if not_scored := score_run(database_url, date_str, detections, dates, args.archive_dir):
                    problems.append(f"{date_str} not scored for {', '.join(not_scored)}")
            except Exception:
                log.exception(f"Could not score {date_str}")
                problems.append(f"{date_str} not scored")
    else:
        log.warning("DATABASE_URL is not set: this run is only in the file")
    log.info(f"Wrote {len(collection['features'])} areas for {day:%Y-%m-%d} to {args.out}")
    if os.environ.get("BACKUP_DIR") and os.environ.get("DATABASE_URL"):
        try:
            backup(Path(os.environ["BACKUP_DIR"]).expanduser(), os.environ["DATABASE_URL"])
        except Exception:
            log.exception("Backup failed")
            problems.append("backup failed")
    else:
        log.warning("BACKUP_DIR is not set: nothing was backed up")
    return problems


if __name__ == "__main__":
    main()
