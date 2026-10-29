"""
Canadian Wildfire Dataset Assembly Pipeline
============================================
Assembles spatially co-registered 64×64 patches from multiple open data sources:

  BC fire perimeters (SHP/GeoJSON)
  CWFIS active hotspots (CSV)
  ERA5 weather reanalysis (NetCDF)
  CWFIS Fire Weather Index rasters (GeoTIFF)
  CDEM terrain (GeoTIFF) → slope/aspect derived
  MODIS NDVI (HDF/GeoTIFF)
  Canada Landcover 2020 (GeoTIFF)

Output: one .npz file per fire-day patch saved to data/processed/patches/
        with metadata JSON manifest in data/processed/splits/

Patch strategy: 64×64 px @ 1 km centred on active fire perimeter + buffer zone.
This keeps ~10–20% positive pixels (vs ~2–5% for random sampling), improving
training signal while still requiring weighted loss.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.warp
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from shapely.geometry import box

from src.utils.config import CHANNELS, PATCH_SIZE, TARGET_RESOLUTION_M

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic daily fire progression
# ---------------------------------------------------------------------------

def generate_daily_masks(
    final_mask: np.ndarray,          # (H, W) — rasterised final perimeter
    n_days: int,                      # total burn duration in days
    fwi_series: list[float],          # FWI per day (length n_days)
    isi_series: list[float],          # ISI per day
    wind_dir_series: list[float],     # wind direction per day (degrees FROM)
    wind_speed_series: list[float],   # wind speed per day (m/s)
) -> list[np.ndarray]:
    """
    Generate synthetic daily binary burn masks by progressively dilating
    a seed mask toward the final documented fire perimeter.

    Physics:
    - Day 0: 3×3 pixel seed at centroid of final perimeter
    - Each day: anisotropic morphological dilation scaled by FWI + ISI + wind
    - Expansion direction biased toward wind direction
    - Hard ceiling: pixels outside the final perimeter cannot burn
    - Calibrated so the final mask area is reached by day n_days

    Returns list of (H, W) float32 arrays, one per day (day 0 … day n_days).
    """
    try:
        from scipy.ndimage import binary_dilation, label
        import numpy as np
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    H, W = final_mask.shape
    n_final_px = final_mask.sum()
    if n_final_px == 0:
        return [np.zeros((H, W), dtype=np.float32)] * (n_days + 1)

    # Seed: 3×3 block at centroid of final perimeter
    ys, xs = np.where(final_mask > 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    seed = np.zeros((H, W), dtype=bool)
    seed[max(0, cy-1):min(H, cy+2), max(0, cx-1):min(W, cx+2)] = True
    seed &= final_mask.astype(bool)
    if not seed.any():
        seed[cy, cx] = True

    masks: list[np.ndarray] = [seed.astype(np.float32)]
    current = seed.copy()

    for day in range(n_days):
        fwi  = float(fwi_series[day])  if day < len(fwi_series)  else 15.0
        isi  = float(isi_series[day])  if day < len(isi_series)  else 6.0
        wdir = float(wind_dir_series[day]) if day < len(wind_dir_series) else 270.0
        wspd = float(wind_speed_series[day]) if day < len(wind_speed_series) else 4.0

        # Expansion radius (pixels/day at 1 km resolution)
        fwi_factor  = np.log1p(max(fwi, 1))  / np.log1p(20)   # norm to FWI=20
        wind_factor = np.log1p(max(wspd, 0))  / np.log1p(8)    # norm to 8 m/s
        isi_factor  = np.log1p(max(isi, 1))   / np.log1p(10)   # norm to ISI=10
        r_along = max(0.4, 2.0 * fwi_factor * (1.0 + 0.7 * isi_factor) * (1.0 + 0.4 * wind_factor))
        r_cross = r_along * 0.4   # cross-wind spread is narrower

        # Build wind-biased elliptical structuring element
        se_size = max(3, int(np.ceil(r_along * 2)) * 2 + 1)
        center  = se_size // 2
        spread_dir = np.radians((wdir + 180) % 360)  # fire spreads INTO wind
        cos_d, sin_d = np.cos(spread_dir), np.sin(spread_dir)

        se = np.zeros((se_size, se_size), dtype=bool)
        for i in range(se_size):
            for j in range(se_size):
                dy, dx = i - center, j - center
                # Rotate to spread direction frame
                d_along =  dx * sin_d + dy * cos_d
                d_cross =  dx * cos_d - dy * sin_d
                if (d_along / r_along) ** 2 + (d_cross / r_cross) ** 2 <= 1.0:
                    se[i, j] = True
        if not se.any():
            se[center, center] = True

        expanded = binary_dilation(current, structure=se)
        # Hard ceiling: cannot burn outside final perimeter
        expanded &= final_mask.astype(bool)
        # If fully saturated already, stay at final mask
        if not expanded.any():
            expanded = current.copy()

        current = expanded
        masks.append(current.astype(np.float32))

    # Ensure last mask equals final perimeter
    masks[-1] = final_mask.astype(np.float32)
    return masks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BC bounding box in EPSG:4326 (lon_min, lat_min, lon_max, lat_max)
BC_BBOX_WGS84 = (-139.1, 48.2, -114.0, 60.0)

# Output CRS — BC Albers (EPSG:3005): equal-area, metres, designed for BC
OUTPUT_CRS = CRS.from_epsg(3005)

# Patch extent in metres at 1 km resolution
PATCH_EXTENT_M = PATCH_SIZE * TARGET_RESOLUTION_M  # 64 000 m = 64 km

# Channel name → variable name in the CEMS fire danger NetCDF (fwi/cems_fwi_YYYY.nc)
CEMS_FWI_VARS = {
    "FFMC": "ffmcode", "DMC": "dufmcode", "DC": "drtcode",
    "ISI": "infsinx", "BUI": "fbupinx", "FWI": "fwinx",
}

# Buffer around fire perimeter edge for patch sampling (metres)
PATCH_BUFFER_M = 5_000   # 5 km beyond active perimeter


# ---------------------------------------------------------------------------
# Patch metadata record
# ---------------------------------------------------------------------------

@dataclass
class PatchMeta:
    patch_id: str          # "{fire_id}_{date}_{row}_{col}"
    fire_id: str
    date: str              # "YYYY-MM-DD"
    year: int
    province: str
    centre_lon: float
    centre_lat: float
    centre_x: float        # Albers easting
    centre_y: float        # Albers northing
    n_burn_pixels: int
    n_total_pixels: int
    burn_fraction: float


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------

def reproject_to_patch_grid(
    src_path: Path,
    transform: rasterio.transform.Affine,
    crs: CRS,
    shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
    band: int = 1,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Reproject and resample a raster band to a target grid.

    Parameters
    ----------
    src_path : Path
        Source raster file.
    transform : Affine
        Target affine transform.
    crs : CRS
        Target CRS (EPSG:3005).
    shape : tuple[int, int]
        (height, width) of target grid.
    resampling : Resampling
        Rasterio resampling method.
    band : int
        Band index (1-based).
    fill_value : float
        NoData fill value.

    Returns
    -------
    np.ndarray, shape (H, W), dtype float32
    """
    destination = np.full(shape, fill_value, dtype=np.float32)
    with rasterio.open(src_path) as src:
        rasterio.warp.reproject(
            source=rasterio.band(src, band),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=fill_value,
        )
    return destination


def burn_mask_from_perimeters(
    perimeters_gdf: gpd.GeoDataFrame,
    date: str,
    transform: rasterio.transform.Affine,
    shape: tuple[int, int],
    crs: CRS,
) -> np.ndarray:
    """
    Rasterise fire perimeter polygons for a given date into a binary mask.

    Pixels that fall inside any fire perimeter active on `date` are set to 1.

    Parameters
    ----------
    perimeters_gdf : GeoDataFrame
        Must have columns: geometry, FIRE_YEAR, IGN_DATE, OUT_DATE (or similar).
        Should already be filtered to the fire year of interest.
    date : str
        "YYYY-MM-DD" — the observation date.
    transform : Affine
        Target patch affine transform.
    shape : tuple[int, int]
        (H, W).
    crs : CRS
        Target CRS.

    Returns
    -------
    np.ndarray, shape (H, W), dtype float32 — values 0.0 or 1.0
    """
    observation = pd.Timestamp(date)

    # Filter to fires that were active on this date
    active = perimeters_gdf[
        (perimeters_gdf["IGN_DATE"] <= observation)
        & (
            perimeters_gdf["OUT_DATE"].isna()
            | (perimeters_gdf["OUT_DATE"] >= observation)
        )
    ].to_crs(crs)

    mask = np.zeros(shape, dtype=np.float32)
    if active.empty:
        return mask

    shapes = [(geom, 1.0) for geom in active.geometry if geom is not None]
    if not shapes:
        return mask

    rasterio.features.rasterize(
        shapes=shapes,
        out=mask,
        transform=transform,
        all_touched=False,
        dtype=np.float32,
    )
    return mask


# ---------------------------------------------------------------------------
# Terrain derivation
# ---------------------------------------------------------------------------

def compute_slope_aspect(elevation: np.ndarray, resolution_m: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees, 0=N clockwise) from an
    elevation array using finite differences.

    Parameters
    ----------
    elevation : np.ndarray, shape (H, W)
    resolution_m : float
        Pixel size in metres.

    Returns
    -------
    slope : np.ndarray, shape (H, W), float32
    aspect : np.ndarray, shape (H, W), float32
    """
    # Gradient in y (north-south) and x (east-west)
    dy, dx = np.gradient(elevation.astype(np.float64), resolution_m)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    aspect_rad = np.arctan2(-dx, dy)          # 0 = north, increases clockwise
    aspect_deg = (np.degrees(aspect_rad) % 360).astype(np.float32)

    return slope_deg, aspect_deg


# ---------------------------------------------------------------------------
# Patch extractor
# ---------------------------------------------------------------------------

class PatchExtractor:
    """
    Extracts 64×64 px patches centred on active fire fronts.

    For each fire-day combination:
      1. Identify the fire perimeter centroid (or grid of centroids for large fires)
      2. Snap to the 1 km output grid
      3. Stack all 18 input channels + target FireMask
      4. Save as .npz with PatchMeta

    Parameters
    ----------
    raw_dir : Path
        Root of data/raw/.
    output_dir : Path
        data/processed/patches/
    province : str
        Two-letter province code ("BC").
    """

    def __init__(self, raw_dir: Path, output_dir: Path, province: str = "BC"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.province = province

        # Derived directories
        self.perimeters_dir = self.raw_dir / "bc_perimeters"
        self.era5_dir = self.raw_dir / "era5"
        self.fwi_dir = self.raw_dir / "fwi"
        self.cdem_path = self.raw_dir / "cdem" / "cdem_bc_srtm90.tif"
        self.ndvi_dir = self.raw_dir / "modis_ndvi"
        self.landcover_path = self.raw_dir / "landcover" / "landcover-2020-classification.tif"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, years: list[int]) -> list[PatchMeta]:
        """
        Run the full assembly pipeline for the given years.

        Returns a list of PatchMeta records for all saved patches.
        """
        all_meta: list[PatchMeta] = []

        perimeters = self._load_perimeters()

        for year in years:
            log.info(f"Assembling patches for {self.province} {year}")
            year_meta = self._process_year(perimeters, year)
            all_meta.extend(year_meta)
            log.info(f"  → {len(year_meta)} patches saved for {year}")

        return all_meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_perimeters(self) -> gpd.GeoDataFrame:
        """
        Load and normalise BC fire perimeter polygons.

        BC Open Government WFS columns (confirmed 2025-09):
          FIRE_NUMBER, FIRE_YEAR, FIRE_DATE, FIRE_SIZE_HECTARES,
          FIRE_CAUSE, FIRE_LABEL, geometry

        We normalise to a consistent internal schema:
          FIRE_ID   ← FIRE_NUMBER
          FIRE_YEAR ← FIRE_YEAR
          IGN_DATE  ← FIRE_DATE (ignition / start date)
          OUT_DATE  ← inferred (FIRE_DATE + 90 days, capped at Oct 31)
          SIZE_HA   ← FIRE_SIZE_HECTARES
        """
        # Prefer the filtered 2012-2023 CNFDB file if available
        cnfdb = self.perimeters_dir / "bc_fire_perimeters_2012_2023.geojson"
        fallback = list(self.perimeters_dir.glob("*.shp")) + list(
            self.perimeters_dir.glob("*.geojson")
        )
        fallback = [f for f in fallback if f != cnfdb]

        if cnfdb.exists():
            candidates = [cnfdb]
            log.info(f"  Using CNFDB filtered file: {cnfdb.name}")
        elif fallback:
            candidates = fallback
            log.info(f"  Using fallback perimeter files: {[f.name for f in candidates]}")
        else:
            raise FileNotFoundError(
                f"No perimeter files found in {self.perimeters_dir}. "
                "Run scripts/download_canada.py first."
            )

        gdfs = [gpd.read_file(f) for f in candidates]
        gdf = pd.concat(gdfs, ignore_index=True)
        gdf = gdf.to_crs("EPSG:4326")

        # Map column names → internal names
        # Handles both BC WFS schema and CNFDB (NRCan) schema
        # CNFDB columns: FIRE_ID, YEAR, MONTH, DAY, REP_DATE, SRC_AGENCY, SIZE_HA, CAUSE
        # BC WFS columns: FIRE_NUMBER, FIRE_YEAR, FIRE_DATE, FIRE_SIZE_HECTARES, FIRE_CAUSE
        rename = {}
        for col in gdf.columns:
            up = col.upper()
            # Fire ID
            if up in ("FIRE_NUMBER", "NID") and "FIRE_ID" not in rename.values():
                rename[col] = "FIRE_ID"
            # Year
            elif up == "YEAR" and "FIRE_YEAR" not in rename.values():
                rename[col] = "FIRE_YEAR"
            elif up == "FIRE_YEAR" and "FIRE_YEAR" not in rename.values():
                rename[col] = "FIRE_YEAR"
            # Ignition date — CNFDB uses REP_DATE, BC WFS uses FIRE_DATE
            elif up in ("REP_DATE", "FIRE_DATE") and "IGN_DATE" not in rename.values():
                rename[col] = "IGN_DATE"
            # Size
            elif up in ("FIRE_SIZE_HECTARES", "GIS_AREA") and "SIZE_HA" not in rename.values():
                rename[col] = "SIZE_HA"
            # Cause
            elif up in ("FIRE_CAUSE", "CAUSE") and "FIRE_CAUSE" not in rename.values():
                rename[col] = "FIRE_CAUSE"

        gdf = gdf.rename(columns=rename)

        # CNFDB: if FIRE_YEAR not yet set, derive from YEAR (already renamed) or IGN_DATE
        if "FIRE_YEAR" not in gdf.columns and "YEAR" in gdf.columns:
            gdf = gdf.rename(columns={"YEAR": "FIRE_YEAR"})

        # CNFDB: construct IGN_DATE from YEAR+MONTH+DAY if REP_DATE absent
        if "IGN_DATE" not in gdf.columns and "MONTH" in gdf.columns and "DAY" in gdf.columns:
            gdf["IGN_DATE"] = pd.to_datetime(
                gdf[["FIRE_YEAR", "MONTH", "DAY"]].rename(
                    columns={"FIRE_YEAR": "year", "MONTH": "month", "DAY": "day"}
                ),
                errors="coerce",
            )

        # Ensure required columns exist
        if "FIRE_ID" not in gdf.columns:
            # Fallback: use row index
            gdf["FIRE_ID"] = gdf.index.astype(str)
        if "FIRE_YEAR" not in gdf.columns:
            gdf["FIRE_YEAR"] = None
        if "SIZE_HA" not in gdf.columns:
            gdf["SIZE_HA"] = 0.0

        # Parse dates
        gdf["IGN_DATE"] = pd.to_datetime(gdf.get("IGN_DATE"), errors="coerce")

        # Infer OUT_DATE: ignition + min(90 days, season end Oct 31)
        def _infer_out_date(row):
            ign = row["IGN_DATE"]
            if pd.isna(ign):
                return pd.NaT
            year = ign.year
            season_end = pd.Timestamp(f"{year}-10-31")
            estimated = ign + pd.Timedelta(days=90)
            return min(estimated, season_end)

        gdf["OUT_DATE"] = gdf.apply(_infer_out_date, axis=1)

        # Infer FIRE_YEAR from IGN_DATE if not set
        mask_no_year = gdf["FIRE_YEAR"].isna()
        gdf.loc[mask_no_year, "FIRE_YEAR"] = gdf.loc[mask_no_year, "IGN_DATE"].dt.year

        # Drop fires with no geometry or no ignition date
        gdf = gdf[gdf.geometry.notna() & gdf["IGN_DATE"].notna()].copy()
        gdf["FIRE_YEAR"] = gdf["FIRE_YEAR"].astype(int)

        log.info(f"Loaded {len(gdf):,} fire perimeters ({gdf['FIRE_YEAR'].min()}–{gdf['FIRE_YEAR'].max()})")
        return gdf

    def _process_year(self, perimeters: gpd.GeoDataFrame, year: int) -> list[PatchMeta]:
        """Extract all patches for one fire season."""
        meta_list: list[PatchMeta] = []

        year_fires = perimeters[perimeters["FIRE_YEAR"] == year].copy()
        if year_fires.empty:
            log.info(f"  No fires found for {year}")
            return meta_list

        # Filter to fires above a minimum size (< 10 ha not useful for spread prediction)
        if "SIZE_HA" in year_fires.columns:
            year_fires = year_fires[year_fires["SIZE_HA"] >= 10.0]

        log.info(f"  {year}: {len(year_fires)} fires ≥ 10 ha")

        for fire_id, fire_group in year_fires.groupby("FIRE_ID", dropna=False):
            fire_id_str = str(fire_id) if fire_id else f"unk_{year}"
            for meta in self._extract_fire_patches(perimeters, fire_group, fire_id_str, year):
                meta_list.append(meta)

        return meta_list

    def _extract_fire_patches(
        self,
        all_perimeters: gpd.GeoDataFrame,
        fire_geom: gpd.GeoDataFrame,
        fire_id: str,
        year: int,
        use_synthetic_progression: bool = True,
    ) -> Iterator[PatchMeta]:
        """
        Yield patches for each day the fire was active.
        For large fires (>100 km²) we tile with overlapping patches.
        """
        if "IGN_DATE" not in fire_geom.columns or fire_geom["IGN_DATE"].isna().all():
            return

        ign_date = fire_geom["IGN_DATE"].dropna().min()
        out_date = (
            fire_geom["OUT_DATE"].dropna().max()
            if "OUT_DATE" in fire_geom.columns
            else pd.Timestamp(f"{year}-11-30")
        )
        if pd.isna(out_date):
            out_date = pd.Timestamp(f"{year}-11-30")

        date_range = list(pd.date_range(ign_date, out_date, freq="D"))
        n_days = len(date_range)
        if n_days < 2:
            return

        # Build patch transform centred on fire centroid (constant across all days)
        import pyproj
        fire_albers = fire_geom.to_crs(OUTPUT_CRS)
        centroid = fire_albers.geometry.unary_union.centroid
        cx, cy = centroid.x, centroid.y
        xmin = cx - PATCH_EXTENT_M / 2
        ymax = cy + PATCH_EXTENT_M / 2
        transform = from_bounds(
            xmin, ymax - PATCH_EXTENT_M, xmin + PATCH_EXTENT_M, ymax,
            PATCH_SIZE, PATCH_SIZE,
        )
        shape = (PATCH_SIZE, PATCH_SIZE)
        transformer = pyproj.Transformer.from_crs(OUTPUT_CRS.to_epsg(), 4326, always_xy=True)
        lon, lat = transformer.transform(cx, cy)

        # Rasterise the full documented fire perimeter (ceiling for synthetic spread)
        full_fire_mask = rasterio.features.rasterize(
            [(geom, 1.0) for geom in fire_albers.geometry if geom is not None],
            out_shape=shape, transform=transform, dtype=np.float32, fill=0.0,
        )
        if full_fire_mask.sum() == 0:
            return  # fire doesn't intersect patch area

        # Build per-day FWI/wind series for physics-driven dilation
        fwi_series, isi_series, wdir_series, wspd_series = [], [], [], []
        for date in date_range:
            fwi_val, isi_val, wdir_val, wspd_val = 15.0, 6.0, 270.0, 4.0
            try:
                # A 64 km patch spans ~2 cells of these 0.25° grids, so the centre value stands in for the patch
                at_centre = lambda arr: float(arr.sel(latitude=lat, longitude=lon, method="nearest"))
                fd = self._fwi_day(date.strftime("%Y-%m-%d"))
                if fd:
                    fwi_val, isi_val = at_centre(fd["FWI"]), at_centre(fd["ISI"])
                ed = self._era5_day(date.strftime("%Y-%m-%d"))
                if ed:
                    u, v = at_centre(ed["u10"]), at_centre(ed["v10"])
                    wspd_val = float(np.hypot(u, v))
                    wdir_val = float(np.degrees(np.arctan2(-u, -v)) % 360)
            except Exception:
                pass
            fwi_series.append(fwi_val)
            isi_series.append(isi_val)
            wdir_series.append(wdir_val)
            wspd_series.append(wspd_val)

        # Generate synthetic daily progression masks
        daily_masks = generate_daily_masks(
            final_mask=full_fire_mask,
            n_days=n_days,
            fwi_series=fwi_series,
            isi_series=isi_series,
            wind_dir_series=wdir_series,
            wind_speed_series=wspd_series,
        )

        # Yield one patch per consecutive day pair (T → T+1)
        for i, date in enumerate(date_range[:-1]):
            date_str = date.strftime("%Y-%m-%d")
            prev_mask = daily_masks[i]       # burn extent at day T (input)
            next_mask = daily_masks[i + 1]   # burn extent at day T+1 (target)

            # Skip uninformative pairs (no new burning)
            delta_px = int((next_mask - prev_mask).clip(0).sum())
            if delta_px == 0 and i > 0:
                continue

            try:
                X = self._build_patch_channels(
                    date_str=date_str,
                    prev_mask=prev_mask,
                    transform=transform,
                    shape=shape,
                    year=year,
                )
            except Exception as exc:
                log.warning(f"Skipping {fire_id} {date_str}: {exc}")
                continue

            patch_id = f"{fire_id}_{date_str}"
            np.savez_compressed(
                self.output_dir / f"{patch_id}.npz",
                X=X,              # (N_CH, 64, 64)
                y=next_mask,      # (64, 64) — next-day burn mask (TARGET)
                transform=np.array(transform),
                crs=OUTPUT_CRS.to_wkt(),
            )

            yield PatchMeta(
                patch_id=patch_id, fire_id=fire_id, date=date_str, year=year,
                province=self.province,
                centre_lon=round(lon, 5), centre_lat=round(lat, 5),
                centre_x=round(cx, 1), centre_y=round(cy, 1),
                n_burn_pixels=int(next_mask.sum()),
                n_total_pixels=int(next_mask.size),
                burn_fraction=round(float(next_mask.sum()) / max(next_mask.size, 1), 4),  # float32 breaks json.dump
            )

    def _build_patch_channels(
        self,
        date_str: str,
        prev_mask: np.ndarray,
        transform: rasterio.transform.Affine,
        shape: tuple[int, int],
        year: int,
    ) -> np.ndarray:
        """
        Build the (N_CH, H, W) input channel array for one patch-day.

        PrevFireMask is provided explicitly (from synthetic daily progression).
        All other channels are loaded from the downloaded data sources.
        """
        H, W = shape
        X = np.zeros((len(CHANNELS), H, W), dtype=np.float32)
        ch = {name: i for i, name in enumerate(CHANNELS)}

        # --- Terrain (static) ---
        if self.cdem_path.exists():
            elev = reproject_to_patch_grid(self.cdem_path, transform, OUTPUT_CRS, shape)
            X[ch["elevation"]] = elev
            slope, aspect = compute_slope_aspect(elev, TARGET_RESOLUTION_M)
            X[ch["slope"]] = slope
            X[ch["aspect"]] = aspect

        # --- ERA5 weather ---
        for var_name, arr in self._load_era5_patch(date_str, transform, shape).items():
            X[ch[var_name]] = arr

        # --- FWI system ---
        for var_name, arr in self._load_fwi_patch(date_str, transform, shape).items():
            X[ch[var_name]] = arr

        # --- NDVI (16-day composites — use most recent available) ---
        ndvi_path = self._find_ndvi(date_str)
        if ndvi_path and ndvi_path.exists():
            X[ch["NDVI"]] = reproject_to_patch_grid(ndvi_path, transform, OUTPUT_CRS, shape)

        # --- Land cover (static) ---
        if self.landcover_path.exists():
            X[ch["landcover"]] = reproject_to_patch_grid(
                self.landcover_path,
                transform,
                OUTPUT_CRS,
                shape,
                resampling=Resampling.mode,  # dominant 30 m class in each 1 km pixel
            )

        # --- Previous fire mask (t=0) — provided by caller (synthetic progression) ---
        X[ch["PrevFireMask"]] = prev_mask

        # BurnP3+ physics prior channel (zero-filled until Phase 2)
        # X[ch["burnp3_burn_prob"]] = 0.0  # already zero from initialization

        return X

    # ------------------------------------------------------------------
    # File finders
    # ------------------------------------------------------------------

    def _find_ndvi(self, date_str: str) -> Optional[Path]:
        """Return the MODIS NDVI composite covering date_str (±16 days)."""
        target = pd.Timestamp(date_str)
        best: Optional[Path] = None
        best_delta = float("inf")
        for f in self.ndvi_dir.glob("*.tif"):
            try:
                stem_date = pd.Timestamp(f.stem[-8:])
                delta = abs((stem_date - target).days)
                if delta < best_delta:
                    best_delta = delta
                    best = f
            except Exception:
                continue
        return best if best_delta <= 16 else None

    @staticmethod
    @functools.lru_cache(maxsize=4)
    def _load_year_nc(path: Path):
        """A year's NetCDF held in memory (~250 MB); lazy reads re-scan the file for every day."""
        import xarray as xr

        return xr.load_dataset(path) if path.exists() else None

    @functools.lru_cache(maxsize=256)
    def _era5_day(self, date_str: str) -> dict:
        """
        BC-wide ERA5 fields for one local day (PDT, UTC-7) from the yearly
        sub-daily files: min/max temperature, minimum relative humidity,
        16:00 wind, and 24 h precipitation. Empty if the files are missing.
        """
        start = pd.Timestamp(date_str) + pd.Timedelta(hours=7)  # 00:00 PDT
        end = start + pd.Timedelta(hours=23)  # still in the same year: seasons end 31 Oct
        inst, accum = (self._load_year_nc(self.era5_dir / f"era5_{start.year}_hourly_{kind}.nc")
                       for kind in ("instant", "accum"))
        if inst is None or accum is None:
            return {}
        inst, accum = inst.sel(valid_time=slice(start, end)), accum.sel(valid_time=slice(start, end))

        t = inst["t2m"] - 273.15
        td = inst["d2m"] - 273.15
        rh = 100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
        afternoon = inst.sel(valid_time=start + pd.Timedelta(hours=16))
        return {
            "temp_max": t.max("valid_time"),
            "temp_min": t.min("valid_time"),
            "humidity": rh.min("valid_time"),
            "u10": afternoon["u10"],
            "v10": afternoon["v10"],
            "precip": accum["tp"].sum("valid_time") * 1000.0,  # m → mm
        }

    def _load_era5_patch(
        self,
        date_str: str,
        transform: rasterio.transform.Affine,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """ERA5 daily fields for date_str, regridded to the patch grid."""
        fields = {
            name: self._regrid_numpy(arr.values.astype(np.float32), arr, transform, shape)
            for name, arr in self._era5_day(date_str).items()
        }
        # Regrid wind as components; interpolating direction breaks across 360° → 0°
        if "u10" in fields:
            u, v = fields.pop("u10"), fields.pop("v10")
            fields["wind_speed"] = np.hypot(u, v)
            fields["wind_dir"] = (np.degrees(np.arctan2(-u, -v)) % 360).astype(np.float32)
        return fields

    @functools.lru_cache(maxsize=256)
    def _fwi_day(self, date_str: str) -> dict:
        """BC-wide CEMS FWI indices for one day. Empty if the year's file or the day is missing."""
        ds = self._load_year_nc(self.fwi_dir / f"cems_fwi_{date_str[:4]}.nc")
        day = pd.Timestamp(date_str)
        if ds is None or day not in ds.indexes["valid_time"]:
            return {}
        # CEMS longitudes run 0–360; the regridder expects -180–180
        day_ds = ds.sel(valid_time=day).assign_coords(longitude=ds.longitude - 360.0)
        return {name: day_ds[var] for name, var in CEMS_FWI_VARS.items()}

    def _load_fwi_patch(
        self,
        date_str: str,
        transform: rasterio.transform.Affine,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """FWI indices for date_str, regridded to the patch grid."""
        return {
            name: self._regrid_numpy(arr.values.astype(np.float32), arr, transform, shape)
            for name, arr in self._fwi_day(date_str).items()
        }

    def _regrid_numpy(
        self,
        data: np.ndarray,
        ds,   # xarray Dataset (for coordinate info)
        transform: rasterio.transform.Affine,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Regrid a (H_src, W_src) numpy array from ERA5 lat/lon grid to
        the target patch grid using rasterio's in-memory reprojection.
        """
        import tempfile
        import xarray as xr

        # Write to a temporary in-memory raster then reproject
        lat_key = next((k for k in ["latitude", "lat"] if k in ds.coords), None)
        lon_key = next((k for k in ["longitude", "lon"] if k in ds.coords), None)
        if lat_key is None or lon_key is None:
            return np.full(shape, 0.0, dtype=np.float32)
        lats = ds.coords[lat_key].values
        lons = ds.coords[lon_key].values

        # Build source affine (ERA5 is on a regular lat/lon grid)
        lon_res = float(lons[1] - lons[0]) if len(lons) > 1 else 0.25
        lat_res = float(lats[1] - lats[0]) if len(lats) > 1 else -0.25

        src_transform = rasterio.transform.from_origin(
            west=float(lons[0]),
            north=float(lats[0]),
            xsize=abs(lon_res),
            ysize=abs(lat_res),
        )
        src_crs = CRS.from_epsg(4326)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            d = data if data.ndim == 2 else data[0]
            h, w = d.shape
            with rasterio.open(
                tmp_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=1,
                dtype=rasterio.float32,
                crs=src_crs,
                transform=src_transform,
            ) as dst:
                dst.write(d.astype(np.float32), 1)

            return reproject_to_patch_grid(tmp_path, transform, OUTPUT_CRS, shape)
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Split manifest builder
# ---------------------------------------------------------------------------

def build_split_manifests(
    all_meta: list[PatchMeta],
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
    splits_dir: Path,
) -> None:
    """
    Write train/val/test manifest JSONs from the list of PatchMeta records.
    Each manifest is a list of patch_ids.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for meta in all_meta:
        if meta.year in train_years:
            splits["train"].append(asdict(meta))
        elif meta.year in val_years:
            splits["val"].append(asdict(meta))
        elif meta.year in test_years:
            splits["test"].append(asdict(meta))

    for split_name, records in splits.items():
        out = splits_dir / f"{split_name}.json"
        with open(out, "w") as f:
            json.dump(records, f, indent=2)
        log.info(
            f"Split '{split_name}': {len(records)} patches → {out}"
        )


# ---------------------------------------------------------------------------
# Channel statistics
# ---------------------------------------------------------------------------

def compute_channel_stats(
    patches_dir: Path,
    train_manifest: Path,
    stats_dir: Path,
) -> None:
    """
    Compute per-channel mean and std over the training split patches.
    Saves stats/channel_stats.json.

    Uses Welford's online algorithm to avoid loading all patches into RAM.
    """
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    with open(train_manifest) as f:
        train_records = json.load(f)

    n_channels = len(CHANNELS)
    count = np.zeros(n_channels, dtype=np.float64)
    mean = np.zeros(n_channels, dtype=np.float64)
    M2 = np.zeros(n_channels, dtype=np.float64)   # sum of squared deviations

    for record in train_records:
        npz_path = patches_dir / f"{record['patch_id']}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        X = data["X"].astype(np.float64)  # (C, H, W)

        for c in range(n_channels):
            pixels = X[c].ravel()
            for val in pixels:
                count[c] += 1
                delta = val - mean[c]
                mean[c] += delta / count[c]
                delta2 = val - mean[c]
                M2[c] += delta * delta2

    std = np.sqrt(np.where(count > 1, M2 / (count - 1), 0.0))
    for i in np.flatnonzero(std == 0):
        log.error(f"Channel '{CHANNELS[i]}' is constant across the training split; its source data was not loaded")

    stats = {
        ch: {"mean": float(mean[i]), "std": float(std[i])}
        for i, ch in enumerate(CHANNELS)
    }
    out = stats_dir / "channel_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Channel stats saved → {out}")
