#!/usr/bin/env python3
"""
Canadian Wildfire Data Downloader
===================================
Downloads all raw data required for Phase 1 of the Canadian Wildfire Spread
Prediction benchmark:

  1. BC historical fire perimeters  (Open Government BC)
  2. NASA FIRMS active-fire hotspots (VIIRS 375 m, BC bounding box)
  3. ERA5 weather reanalysis         (Copernicus CDS)
  4. ERA5-HRS FWI indices            (Zenodo, McElhinny et al. 2020)
     — overwintered DC recommended for western Canada
  5. CWFIS operational FWI           (NRCan datamart, 2019–2023 fallback)
  6. Canadian Digital Elevation Model (CDEM, NRCan)
  7. Canada Land Cover 2020          (NRCan)
  8. MODIS MOD13A2 NDVI              (NASA LP DAAC via earthaccess)

Usage
-----
  python scripts/download_canada.py --province BC --years 2012 2023 [--data-dir data/raw]

Notes
-----
- ERA5 download requires a ~/.cdsapirc file with a Copernicus CDS API key.
  Register at: https://cds.climate.copernicus.eu
- FIRMS download requires a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/
  Set it as environment variable: export FIRMS_MAP_KEY=your_key_here
- MODIS download uses NASA earthaccess (free account required):
  https://urs.earthdata.nasa.gov/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Bounding box for British Columbia
# ---------------------------------------------------------------------------
BC_BBOX = {
    "lon_min": -139.1,
    "lat_min": 48.2,
    "lon_max": -114.0,
    "lat_max": 60.0,
}

# ---------------------------------------------------------------------------
# ERA5-HRS FWI Zenodo DOIs (overwintered DC — recommended for western Canada)
# McElhinny et al. 2020, doi:10.5281/zenodo.3626193
# ---------------------------------------------------------------------------
FWI_ZENODO_DOIS_OVERWINTERED = {
    "FFMC": "10.5281/zenodo.3540922",
    "DMC":  "10.5281/zenodo.3540924",
    "DC":   "10.5281/zenodo.3540926",
    "ISI":  "10.5281/zenodo.3540920",
    "BUI":  "10.5281/zenodo.3540918",
    "FWI":  "10.5281/zenodo.3539654",
    "DSR":  "10.5281/zenodo.3540928",
}

ZENODO_API = "https://zenodo.org/api/records"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, desc: str = "", chunk_size: int = 8192) -> Path:
    """Stream-download a file with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info(f"  already exists, skipping: {dest.name}")
        return dest

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        desc=desc or dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            bar.update(size)

    return dest


def _zenodo_files(doi: str) -> list[dict]:
    """Return file metadata list for a Zenodo record identified by its DOI."""
    record_id = doi.split("/")[-1]
    resp = requests.get(f"{ZENODO_API}/{record_id}", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("files", [])


# ---------------------------------------------------------------------------
# 1. BC Fire Perimeters
# ---------------------------------------------------------------------------

def download_bc_perimeters(dest_dir: Path) -> None:
    """
    Download BC historical wildfire perimeters from the BC Data Catalogue.

    Source: BC Wildfire Service — Historical Fire Perimeters
    URL:    https://catalogue.data.gov.bc.ca/dataset/fire-perimeters-historical
    Format: GeoJSON (WGS84) or Shapefile
    License: Open Government Licence – BC
    """
    log.info("=== BC Fire Perimeters ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Direct GeoJSON download via BC Data Catalogue BCGW
    # The WFS endpoint returns all historical perimeters
    base_url = (
        "https://openmaps.gov.bc.ca/geo/pub/WHSE_LAND_AND_NATURAL_RESOURCE"
        ".PROT_HISTORICAL_FIRE_POLYS_SP/ows"
    )
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "WHSE_LAND_AND_NATURAL_RESOURCE:PROT_HISTORICAL_FIRE_POLYS_SP",
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
    }

    param_str = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base_url}?{param_str}"
    out = dest_dir / "bc_fire_perimeters_historical.geojson"

    if out.exists():
        log.info(f"  already exists: {out.name}")
        return

    log.info(f"  Downloading BC perimeters (this may be large)…")
    try:
        _download_file(url, out, "BC fire perimeters")
        log.info(f"  Saved → {out}")
    except Exception as exc:
        log.error(f"  BC perimeters download failed: {exc}")
        log.info("  Fallback: download manually from https://catalogue.data.gov.bc.ca/"
                 "dataset/fire-perimeters-historical and place in data/raw/bc_perimeters/")


# ---------------------------------------------------------------------------
# 2. NASA FIRMS Active-Fire Hotspots
# ---------------------------------------------------------------------------

def download_firms_hotspots(
    dest_dir: Path,
    years: list[int],
    map_key: Optional[str] = None,
) -> None:
    """
    Download VIIRS 375m active-fire observations for BC from NASA FIRMS.

    Requires a free FIRMS MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/
    Set env var FIRMS_MAP_KEY or pass --firms-key.

    Format: CSV, one file per year
    """
    log.info("=== NASA FIRMS VIIRS 375m hotspots ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    key = map_key or os.environ.get("FIRMS_MAP_KEY", "")
    if not key:
        log.warning(
            "  FIRMS_MAP_KEY not set. Register at https://firms.modaps.eosdis.nasa.gov/api/ "
            "and export FIRMS_MAP_KEY=your_key"
        )
        return

    bb = BC_BBOX
    area = f"{bb['lon_min']},{bb['lat_min']},{bb['lon_max']},{bb['lat_max']}"

    for year in years:
        out = dest_dir / f"viirs_375m_bc_{year}.csv"
        if out.exists():
            log.info(f"  already exists: {out.name}")
            continue

        # FIRMS area API: annual archive
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}"
            f"/VIIRS_SNPP_NRT/{area}/1/{year}-01-01"
        )
        try:
            _download_file(url, out, f"FIRMS VIIRS {year}")
            log.info(f"  Saved → {out}")
        except Exception as exc:
            log.warning(f"  FIRMS {year} failed: {exc}")


# ---------------------------------------------------------------------------
# 3. ERA5 Weather Reanalysis
# ---------------------------------------------------------------------------

def download_era5(dest_dir: Path, years: list[int]) -> None:
    """
    Download daily ERA5 surface variables for BC from the Copernicus CDS.

    Variables:
      - 10m_u_component_of_wind  (u10)
      - 10m_v_component_of_wind  (v10)
      - 2m_temperature           (t2m) — used for temp_min, temp_max
      - 2m_dewpoint_temperature  (d2m) — used to derive relative humidity
      - total_precipitation      (tp)

    Requires ~/.cdsapirc with UID and API key from https://cds.climate.copernicus.eu

    Format: NetCDF, one file per year-month
    """
    log.info("=== ERA5 weather reanalysis ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cdsapi
    except ImportError:
        log.error("  cdsapi not installed. Run: pip install cdsapi")
        return

    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        log.warning(
            "  ~/.cdsapirc not found. Register at https://cds.climate.copernicus.eu "
            "and create this file with your API key."
        )
        return

    c = cdsapi.Client(quiet=True)
    bb = BC_BBOX

    fire_months = [str(m).zfill(2) for m in range(4, 11)]  # April–October

    for year in years:
        out = dest_dir / f"era5_{year}.nc"
        if out.exists():
            log.info(f"  already exists: {out.name}")
            continue

        log.info(f"  Requesting ERA5 {year} (fire season months Apr–Oct)…")
        try:
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "2m_temperature",
                        "2m_dewpoint_temperature",
                        "total_precipitation",
                    ],
                    "year": str(year),
                    "month": fire_months,
                    "day": [str(d).zfill(2) for d in range(1, 32)],
                    "time": "12:00",     # daily noon snapshot
                    "area": [
                        bb["lat_max"], bb["lon_min"],
                        bb["lat_min"], bb["lon_max"],
                    ],
                    "format": "netcdf",
                },
                str(out),
            )
            log.info(f"  Saved → {out}")
        except Exception as exc:
            log.error(f"  ERA5 {year} failed: {exc}")


# ---------------------------------------------------------------------------
# 4. ERA5-HRS FWI Indices (Zenodo, McElhinny et al. 2020)
# ---------------------------------------------------------------------------

def download_fwi_zenodo(
    dest_dir: Path,
    years: list[int],
    variables: Optional[list[str]] = None,
) -> None:
    """
    Download ERA5-HRS Fire Weather Index system indices from Zenodo.

    Source: McElhinny et al. 2020, doi:10.5281/zenodo.3626193
    Uses the overwintered DC version (recommended for western Canada).
    Coverage: 1979–2018 (daily, global, NetCDF).

    For years > 2018 (i.e., 2019–2023 test split), fall back to CWFIS
    operational data (see download_cwfis_fwi).

    Variables: FFMC, DMC, DC, ISI, BUI, FWI (DSR optional)
    """
    log.info("=== ERA5-HRS FWI indices (Zenodo, overwintered DC) ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    if variables is None:
        variables = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

    zenodo_years = [y for y in years if y <= 2018]
    if not zenodo_years:
        log.info("  No Zenodo-covered years (≤2018) in request; skipping.")
        return

    for var in variables:
        doi = FWI_ZENODO_DOIS_OVERWINTERED.get(var)
        if not doi:
            log.warning(f"  No DOI found for {var}, skipping.")
            continue

        var_dir = dest_dir / var.lower()
        var_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"  Fetching file list for {var} (DOI: {doi})…")
        try:
            files = _zenodo_files(doi)
        except Exception as exc:
            log.error(f"  Could not fetch Zenodo record for {var}: {exc}")
            continue

        for file_info in files:
            fname: str = file_info.get("key", "")
            fsize: int = file_info.get("size", 0)
            furl: str = file_info.get("links", {}).get("self", "")

            # Files are named like "FFMC_1979.nc", "FFMC_1980.nc", …
            # Filter to only the years we need
            try:
                file_year = int(fname.replace(".nc", "").split("_")[-1])
            except ValueError:
                continue
            if file_year not in zenodo_years:
                continue

            out = var_dir / fname
            if out.exists():
                log.info(f"    already exists: {fname}")
                continue

            log.info(f"    Downloading {fname} ({fsize / 1e6:.1f} MB)…")
            try:
                _download_file(furl, out, fname)
            except Exception as exc:
                log.error(f"    {fname} failed: {exc}")

    log.info("  ERA5-HRS FWI download complete.")


# ---------------------------------------------------------------------------
# 5. CWFIS Operational FWI (2019–2023 fallback)
# ---------------------------------------------------------------------------

def download_cwfis_fwi(dest_dir: Path, years: list[int]) -> None:
    """
    Download operational FWI rasters from the NRCan CWFIS Datamart.

    Used for years not covered by the Zenodo ERA5-HRS dataset (i.e., >2018).
    Format: GeoTIFF, one file per day per variable.

    CWFIS Datamart: https://cwfis.cfs.nrcan.gc.ca/datamart
    """
    log.info("=== CWFIS operational FWI (2019+) ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    cwfis_years = [y for y in years if y > 2018]
    if not cwfis_years:
        log.info("  No post-2018 years requested; CWFIS fallback not needed.")
        return

    base_url = "https://cwfis.cfs.nrcan.gc.ca/datamart/fwi"
    variables = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]

    import pandas as pd

    for year in cwfis_years:
        dates = pd.date_range(f"{year}-04-01", f"{year}-10-31", freq="D")
        for date in tqdm(dates, desc=f"CWFIS FWI {year}"):
            date_str = date.strftime("%Y%m%d")
            for var in variables:
                out = dest_dir / var.lower() / f"{var}_{date_str}.tif"
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                url = f"{base_url}/{var.lower()}/{date_str}/{var}_{date_str}.tif"
                try:
                    _download_file(url, out)
                except Exception:
                    pass  # Many days may be missing; silent skip


# ---------------------------------------------------------------------------
# 6. Canadian Digital Elevation Model (CDEM)
# ---------------------------------------------------------------------------

def download_cdem(dest_dir: Path) -> None:
    """
    Download the Canadian Digital Elevation Model at 1 arc-second (~30m).

    Source: NRCan Open Government
    The full national mosaic is large (~12 GB). We download BC-relevant tiles
    or the national mosaic index and then retrieve only BC tiles.

    For convenience, we provide the 1-km pre-mosaicked version URL when available.
    Manual alternative: https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333
    """
    log.info("=== Canadian Digital Elevation Model (CDEM) ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # NRCan CDEM mosaic for BC — GeoTIFF
    # Tile index: https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/
    ftp_base = "https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec"

    # BC spans NTS sheets 82, 83, 92, 93, 94, 102, 103, 104
    bc_sheets = ["082", "083", "092", "093", "094", "102", "103", "104"]

    for sheet in bc_sheets:
        fname = f"cdem_dem_{sheet}_tif.zip"
        out_zip = dest_dir / fname
        out_dir = dest_dir / sheet

        if out_dir.exists():
            log.info(f"  already extracted: {sheet}")
            continue

        url = f"{ftp_base}/{fname}"
        try:
            _download_file(url, out_zip, f"CDEM {sheet}")
            with zipfile.ZipFile(out_zip, "r") as z:
                z.extractall(out_dir)
            out_zip.unlink()
            log.info(f"  Extracted → {out_dir}")
        except Exception as exc:
            log.warning(f"  CDEM {sheet} failed: {exc}")


# ---------------------------------------------------------------------------
# 7. Canada Land Cover 2020
# ---------------------------------------------------------------------------

def download_landcover(dest_dir: Path) -> None:
    """
    Download the Canada Land Cover 2020 product from NRCan.

    Source: https://open.canada.ca/data/en/dataset/4e615eae-b90c-420b-adee-2ca35896caf6
    Resolution: 30 m, GeoTIFF
    License: Open Government Licence – Canada
    """
    log.info("=== Canada Land Cover 2020 ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    out = dest_dir / "canada_landcover_2020_30m.zip"
    extracted = dest_dir / "canada_lc_2020"

    if extracted.exists():
        log.info("  already extracted.")
        return

    # NRCan STAC / direct download URL
    url = (
        "https://ftp.maps.canada.ca/pub/nrcan_rncan/Land-cover_Couverture-du-sol"
        "/canada-landcover_canada-couverture-du-sol/CanadaLandcover2020.zip"
    )
    try:
        _download_file(url, out, "Canada Landcover 2020")
        with zipfile.ZipFile(out, "r") as z:
            z.extractall(extracted)
        out.unlink()
        log.info(f"  Extracted → {extracted}")
    except Exception as exc:
        log.error(f"  Landcover download failed: {exc}")
        log.info(
            "  Manual download: https://open.canada.ca/data/en/dataset/"
            "4e615eae-b90c-420b-adee-2ca35896caf6"
        )


# ---------------------------------------------------------------------------
# 8. MODIS MOD13A2 NDVI
# ---------------------------------------------------------------------------

def download_modis_ndvi(dest_dir: Path, years: list[int]) -> None:
    """
    Download MODIS MOD13A2 16-day NDVI composites for BC tiles.

    Requires a free NASA Earthdata account: https://urs.earthdata.nasa.gov/
    Uses the `earthaccess` Python package for authenticated downloads.

    BC spans MODIS tiles: h09v02, h09v03, h10v02, h10v03
    """
    log.info("=== MODIS MOD13A2 NDVI ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        import earthaccess
    except ImportError:
        log.warning(
            "  earthaccess not installed. Run: pip install earthaccess\n"
            "  Then register at https://urs.earthdata.nasa.gov/"
        )
        return

    bc_tiles = ["h09v02", "h09v03", "h10v02", "h10v03"]

    try:
        earthaccess.login(strategy="netrc")
    except Exception:
        log.warning(
            "  earthaccess login failed. Create ~/.netrc with your NASA Earthdata credentials:\n"
            "    machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASS"
        )
        return

    for year in years:
        results = earthaccess.search_data(
            short_name="MOD13A2",
            version="061",
            temporal=(f"{year}-04-01", f"{year}-10-31"),
            granule_name=[f"*{tile}*" for tile in bc_tiles],
        )
        if not results:
            log.warning(f"  No MOD13A2 results for {year}")
            continue

        year_dir = dest_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        log.info(f"  Downloading {len(results)} MOD13A2 granules for {year}…")
        earthaccess.download(results, str(year_dir))

    log.info("  MODIS NDVI download complete.")


# ---------------------------------------------------------------------------
# BurnP3+ setup instructions
# ---------------------------------------------------------------------------

def print_burnp3_instructions() -> None:
    """
    Print setup instructions for BurnP3+ (not auto-downloaded — requires
    SyncroSim installation).
    """
    print("""
=== BurnP3+ Physics-Based Fire Growth Model ===

BurnP3+ (Canadian Forest Service) provides Monte Carlo fire spread simulation
using the FBP system. It is the Canadian equivalent of FireBench.

It is NOT downloaded automatically — it requires SyncroSim platform installation.

Setup:
  1. Install SyncroSim (Windows/Linux): https://syncrosim.com/download/
  2. Install BurnP3+ package:
       pip install pysyncrosim
       import syncrosim as ss
       ss.package.install("burnp3")
  3. Install a fire growth engine:
       - FireSTARR: https://github.com/CWFMF/FireSTARR
       - Prometheus: https://firegrowthmodel.ca/#/prometheus_software

Integration plan (Phase 2):
  - Use BurnP3+ via pysyncrosim to run 100s of fire scenarios per historical
    weather year, varying ignition location and weather within FWI uncertainty
  - Use simulated burn probability rasters as:
    (a) Additional training samples (augmented dataset)
    (b) Physics-informed prior feature channel ("burnp3_burn_prob")
  - This enables the real-data + simulated-data foundation model training track

Reference: https://burnp3.github.io/BurnP3Plus/
Discord:   https://discord.gg/76QzY8eAYr
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all Canadian wildfire data for Phase 1"
    )
    parser.add_argument(
        "--province", default="BC", choices=["BC"], help="Province (BC for Phase 1)"
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=[2012, 2023],
        help="Year range (inclusive). Default: 2012 2023",
    )
    parser.add_argument(
        "--data-dir", default="data/raw", help="Root directory for raw data"
    )
    parser.add_argument(
        "--firms-key", default=None, help="NASA FIRMS MAP_KEY (or set FIRMS_MAP_KEY env var)"
    )
    parser.add_argument(
        "--skip-era5", action="store_true", help="Skip ERA5 download (requires CDS API key)"
    )
    parser.add_argument(
        "--skip-modis", action="store_true", help="Skip MODIS NDVI download (requires Earthdata)"
    )
    parser.add_argument(
        "--burnp3-info", action="store_true", help="Print BurnP3+ setup instructions and exit"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.burnp3_info:
        print_burnp3_instructions()
        sys.exit(0)

    years = list(range(args.years[0], args.years[1] + 1))
    data_dir = Path(args.data_dir)

    log.info(f"Province: {args.province} | Years: {years[0]}–{years[-1]}")
    log.info(f"Data directory: {data_dir.resolve()}")

    # 1. BC fire perimeters
    download_bc_perimeters(data_dir / "bc_perimeters")

    # 2. FIRMS hotspots
    download_firms_hotspots(data_dir / "cwfis_hotspots", years, args.firms_key)

    # 3. ERA5 weather
    if not args.skip_era5:
        download_era5(data_dir / "era5", years)
    else:
        log.info("=== ERA5: skipped ===")

    # 4. ERA5-HRS FWI (Zenodo, 2012–2018)
    download_fwi_zenodo(data_dir / "fwi", years)

    # 5. CWFIS operational FWI (2019–2023 fallback)
    download_cwfis_fwi(data_dir / "fwi", years)

    # 6. CDEM
    download_cdem(data_dir / "cdem")

    # 7. Canada Landcover 2020
    download_landcover(data_dir / "landcover")

    # 8. MODIS NDVI
    if not args.skip_modis:
        download_modis_ndvi(data_dir / "modis_ndvi", years)
    else:
        log.info("=== MODIS NDVI: skipped ===")

    log.info("\n✓ All downloads complete. Run next:")
    log.info("  python scripts/assemble_patches.py --province BC")
    log.info("\nFor BurnP3+ setup (Phase 2 simulation augmentation):")
    log.info("  python scripts/download_canada.py --burnp3-info")


if __name__ == "__main__":
    main()
