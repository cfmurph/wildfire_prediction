#!/usr/bin/env python3
"""
Patch Assembly Script
======================
Runs the full data assembly pipeline:
  1. Load all raw downloaded data
  2. Co-register all layers to a 1 km BC Albers grid
  3. Extract 64×64 patches centred on active fire perimeters
  4. Save as .npz files to data/processed/patches/
  5. Build train/val/test split manifests
  6. Compute per-channel statistics from the training split

Usage:
  python scripts/assemble_patches.py --province BC [--data-dir data/raw]

Expected raw data structure (after running download_canada.py):
  data/raw/bc_perimeters/*.geojson or *.shp
  data/raw/era5/era5_YYYY.nc
  data/raw/fwi/{FFMC,DMC,DC,ISI,BUI,FWI}/*.nc  (Zenodo ERA5-HRS)
  data/raw/fwi/{FFMC,DMC,DC,ISI,BUI,FWI}/*.tif (CWFIS operational)
  data/raw/cdem/...
  data/raw/modis_ndvi/...
  data/raw/landcover/...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble Canadian wildfire patch dataset")
    p.add_argument("--province", default="BC", choices=["BC"])
    p.add_argument("--data-dir", default="data/raw", help="Raw data root")
    p.add_argument("--output-dir", default="data/processed", help="Processed data root")
    p.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=[2012, 2023],
    )
    p.add_argument("--train-years-end", type=int, default=2020)
    p.add_argument("--val-years", nargs="+", type=int, default=[2021, 2022])
    p.add_argument("--test-years", nargs="+", type=int, default=[2023])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    patches_dir = output_dir / "patches"
    splits_dir = output_dir / "splits"
    stats_dir = output_dir / "stats"

    years = list(range(args.years[0], args.years[1] + 1))
    train_years = [y for y in years if y <= args.train_years_end]
    val_years = args.val_years
    test_years = args.test_years

    log.info(f"Province: {args.province}")
    log.info(f"Years: {years[0]}–{years[-1]}")
    log.info(f"Train: {train_years} | Val: {val_years} | Test: {test_years}")

    # Check raw data exists
    perimeter_candidates = list(raw_dir.glob("bc_perimeters/*.geojson")) + \
                           list(raw_dir.glob("bc_perimeters/*.shp"))
    if not perimeter_candidates:
        log.error(
            "No BC perimeter files found. Run: python scripts/download_canada.py first."
        )
        sys.exit(1)

    # Import here so the script can be run after install
    from src.data.assemble import (
        PatchExtractor,
        build_split_manifests,
        compute_channel_stats,
    )

    # 1. Extract patches
    extractor = PatchExtractor(
        raw_dir=raw_dir,
        output_dir=patches_dir,
        province=args.province,
    )
    all_meta = extractor.run(years=years)
    log.info(f"Total patches extracted: {len(all_meta):,}")

    if not all_meta:
        log.error(
            "No patches were extracted. Check that raw data files are present "
            "and that fire perimeters have correct date columns."
        )
        sys.exit(1)

    # 2. Build split manifests
    build_split_manifests(
        all_meta=all_meta,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        splits_dir=splits_dir,
    )

    # 3. Compute channel statistics on training split
    train_manifest = splits_dir / "train.json"
    compute_channel_stats(
        patches_dir=patches_dir,
        train_manifest=train_manifest,
        stats_dir=stats_dir,
    )

    log.info("\n✓ Assembly complete. Next steps:")
    log.info("  1. jupyter notebook notebooks/01_eda.ipynb")
    log.info("  2. python src/training/train.py --config-name baseline_rf")
    log.info("  3. python src/training/train.py --config-name unet")
    log.info("  4. python src/training/train.py --config-name convlstm")
    log.info("  5. mlflow ui --backend-store-uri experiments/mlruns")


if __name__ == "__main__":
    main()
