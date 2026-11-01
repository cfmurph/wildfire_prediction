"""
Mosaic MODIS MOD13A2 NDVI tiles into one BC GeoTIFF per 16-day composite.

download_canada.py fetches the HDF4 tiles into data/raw/modis_ndvi/<year>/.
This writes data/raw/modis_ndvi/ndvi_<YYYYMMDD>.tif in EPSG:3005 at 1 km,
dated by the composite's LAST day, which is what the assembler matches on.

Needs pyhdf (pip install -e ".[data]"): rasterio's GDAL wheels cannot read HDF4.

Usage:
    python scripts/prepare_ndvi.py
"""

from __future__ import annotations

import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.assemble import BC_BBOX_WGS84, OUTPUT_CRS  # noqa: E402

log = logging.getLogger(__name__)

NDVI_DIR = Path("data/raw/modis_ndvi")

# MODIS sinusoidal grid: 1200 × 1200 px tiles, h counted from the west, v from the north
SINUSOIDAL = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"
TILE_M = 1111950.5196666666
PIXEL_M = TILE_M / 1200
X_MIN, Y_MAX = -20015109.354, 10007554.677
# Tiles covering BC; the download's bounding-box search also returned two far-away tiles
H_RANGE, V_RANGE = range(9, 13), range(2, 5)

NAME = re.compile(r"MOD13A2\.A(\d{4})(\d{3})\.h(\d{2})v(\d{2})\.")


def mosaic(tiles: dict[tuple[int, int], Path]) -> np.ndarray:
    """One composite's BC tiles as a single float32 NDVI array (NaN where missing)."""
    from pyhdf.SD import SD, SDC

    out = np.full((len(V_RANGE) * 1200, len(H_RANGE) * 1200), np.nan, dtype=np.float32)
    for (h, v), path in tiles.items():
        raw = SD(str(path), SDC.READ).select("1 km 16 days NDVI").get()
        r, c = (v - V_RANGE.start) * 1200, (h - H_RANGE.start) * 1200
        out[r : r + 1200, c : c + 1200] = np.where(raw == -3000, np.nan, raw / 10000.0)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    composites: dict[pd.Timestamp, dict] = defaultdict(dict)
    for path in NDVI_DIR.glob("*/MOD13A2.*.hdf"):
        year, doy, h, v = map(int, NAME.search(path.name).groups())
        if h in H_RANGE and v in V_RANGE:
            start = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
            composites[start][h, v] = path

    src_transform = from_origin(
        X_MIN + H_RANGE.start * TILE_M, Y_MAX - V_RANGE.start * TILE_M, PIXEL_M, PIXEL_M
    )
    left, bottom, right, top = transform_bounds("EPSG:4326", OUTPUT_CRS, *BC_BBOX_WGS84)
    dst_transform = from_origin(left, top, 1000, 1000)
    shape = (int((top - bottom) // 1000) + 1, int((right - left) // 1000) + 1)

    full = max(len(tiles) for tiles in composites.values())
    for start, tiles in sorted(composites.items()):
        last_day = start + pd.Timedelta(days=15)
        if len(tiles) < full:
            # The assembler falls back to the previous composite rather than a partial one
            log.warning(f"Skipping composite ending {last_day:%Y-%m-%d}: {len(tiles)}/{full} tiles")
            continue
        dest = NDVI_DIR / f"ndvi_{last_day:%Y%m%d}.tif"
        if dest.exists():
            continue
        bc = np.full(shape, np.nan, dtype=np.float32)
        reproject(
            mosaic(tiles), bc,
            src_transform=src_transform, src_crs=SINUSOIDAL, src_nodata=np.nan,
            dst_transform=dst_transform, dst_crs=OUTPUT_CRS, dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        with rasterio.open(
            dest, "w", driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="float32",
            crs=OUTPUT_CRS, transform=dst_transform, nodata=np.nan, compress="deflate", predictor=3,
        ) as dst:
            dst.write(bc, 1)
        log.info(f"{dest.name}: {np.isfinite(bc).mean():.0%} of the BC box has data")


if __name__ == "__main__":
    main()
