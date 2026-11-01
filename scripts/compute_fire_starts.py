"""
Typical monthly fire-start density across BC, by cause, for the map's Risk view.

Counts BC fire starts in the NFDB point dataset (download_canada.py fetches it) per
month and cause on a 5 km grid, smooths them with a ~10 km Gaussian so a single old
fire does not paint its own hotspot, and averages over the years. Writes
data/processed/stats/fire_starts.json, which the API serves.

Usage:
    python scripts/compute_fire_starts.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

log = logging.getLogger(__name__)

POINTS = sorted(Path("data/raw/ignition/nfdb_points").glob("NFDB_point*.shp"))
OUT = Path("data/processed/stats/fire_starts.json")
YEARS = (1996, 2025)
CELL_M = 5_000
SMOOTH_M = 10_000
PER_KM2 = 1_000
MIN_DENSITY = 0.1  # starts per 1,000 km² per month: about one per decade
CAUSES = {"lightning": "N", "human": "H"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    starts = gpd.read_file(
        POINTS[-1],
        where=f"SRC_AGENCY = 'BC' AND YEAR >= {YEARS[0]} AND YEAR <= {YEARS[1]}",
        columns=["SRC_AGENCY", "YEAR", "MONTH", "CAUSE"],
    ).to_crs("EPSG:3005")
    log.info(f"{len(starts):,} BC fire starts {YEARS[0]}–{YEARS[1]} from {POINTS[-1].name}")

    x, y = starts.geometry.x.to_numpy(), starts.geometry.y.to_numpy()
    x0, y0 = x.min() - 5 * SMOOTH_M, y.min() - 5 * SMOOTH_M
    col, row = ((x - x0) // CELL_M).astype(int), ((y - y0) // CELL_M).astype(int)
    shape = (row.max() + 1 + 5 * SMOOTH_M // CELL_M, col.max() + 1 + 5 * SMOOTH_M // CELL_M)
    rows, cols = np.indices(shape)
    to_wgs84 = Transformer.from_crs("EPSG:3005", "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(x0 + (cols + 0.5) * CELL_M, y0 + (rows + 0.5) * CELL_M)

    n_years = YEARS[1] - YEARS[0] + 1
    per_cell = PER_KM2 / (CELL_M / 1000) ** 2
    months = {}
    for month in range(1, 13):
        months[month] = {}
        for name, code in CAUSES.items():
            sel = (starts["MONTH"].to_numpy() == month) & (starts["CAUSE"].to_numpy() == code)
            counts = np.zeros(shape)
            np.add.at(counts, (row[sel], col[sel]), 1)
            density = gaussian_filter(counts, sigma=SMOOTH_M / CELL_M) / n_years * per_cell
            keep = density >= MIN_DENSITY
            months[month][name] = [
                [round(float(a), 3), round(float(b), 3), round(float(d), 2)]
                for a, b, d in zip(lon[keep], lat[keep], density[keep])
            ]
        log.info(f"  month {month:2d}: " + ", ".join(f"{k} {len(v):,} cells" for k, v in months[month].items()))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "years": list(YEARS),
        "cell_km": CELL_M // 1000,
        "smoothing_km": SMOOTH_M // 1000,
        "unit": f"fire starts per {PER_KM2:,} km² in a typical month",
        "months": months,
    }, separators=(",", ":")))
    log.info(f"Saved → {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
