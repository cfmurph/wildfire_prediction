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
# Constants
# ---------------------------------------------------------------------------

# BC bounding box in EPSG:4326 (lon_min, lat_min, lon_max, lat_max)
BC_BBOX_WGS84 = (-139.1, 48.2, -114.0, 60.0)

# Output CRS — BC Albers (EPSG:3005): equal-area, metres, designed for BC
OUTPUT_CRS = CRS.from_epsg(3005)

# Patch extent in metres at 1 km resolution
PATCH_EXTENT_M = PATCH_SIZE * TARGET_RESOLUTION_M  # 64 000 m = 64 km

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
        self.cdem_path = self.raw_dir / "cdem" / "cdem_bc_1km.tif"
        self.ndvi_dir = self.raw_dir / "modis_ndvi"
        self.landcover_path = self.raw_dir / "landcover" / "canada_lc_2020_bc_1km.tif"

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
        """Load and normalise BC fire perimeter polygons."""
        candidates = list(self.perimeters_dir.glob("*.shp")) + list(
            self.perimeters_dir.glob("*.geojson")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No perimeter files found in {self.perimeters_dir}. "
                "Run scripts/download_canada.py first."
            )

        gdfs = [gpd.read_file(f) for f in candidates]
        gdf = pd.concat(gdfs, ignore_index=True)
        gdf = gdf.to_crs("EPSG:4326")

        # Normalise column names — BC open data uses specific names
        col_map = {}
        for col in gdf.columns:
            low = col.upper()
            if "IGN" in low and "DATE" in low:
                col_map[col] = "IGN_DATE"
            elif "OUT" in low and "DATE" in low:
                col_map[col] = "OUT_DATE"
            elif "FIRE_YEAR" in low or "YEAR" in low:
                col_map[col] = "FIRE_YEAR"
            elif "FIRE_NUM" in low or "FIRE_ID" in low or "ID" in low:
                col_map[col] = "FIRE_ID"
        gdf = gdf.rename(columns=col_map)

        # Parse dates
        for datecol in ["IGN_DATE", "OUT_DATE"]:
            if datecol in gdf.columns:
                gdf[datecol] = pd.to_datetime(gdf[datecol], errors="coerce")

        return gdf

    def _process_year(self, perimeters: gpd.GeoDataFrame, year: int) -> list[PatchMeta]:
        """Extract all patches for one fire season."""
        meta_list: list[PatchMeta] = []

        year_fires = perimeters[perimeters.get("FIRE_YEAR", pd.Series(dtype=int)) == year]
        if year_fires.empty:
            # Try inferring year from IGN_DATE
            year_fires = perimeters[
                perimeters["IGN_DATE"].dt.year == year
            ] if "IGN_DATE" in perimeters.columns else perimeters

        # Iterate over individual fires
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

        date_range = pd.date_range(ign_date, out_date, freq="D")

        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            # Centroid of current perimeter in Albers
            fire_albers = fire_geom.to_crs(OUTPUT_CRS)
            centroid = fire_albers.geometry.unary_union.centroid

            cx, cy = centroid.x, centroid.y

            # Build patch grid transform centred on fire
            xmin = cx - PATCH_EXTENT_M / 2
            ymax = cy + PATCH_EXTENT_M / 2
            transform = from_bounds(
                left=xmin,
                bottom=ymax - PATCH_EXTENT_M,
                right=xmin + PATCH_EXTENT_M,
                top=ymax,
                width=PATCH_SIZE,
                height=PATCH_SIZE,
            )
            shape = (PATCH_SIZE, PATCH_SIZE)

            # Stack channels
            try:
                patch_array, target_array = self._build_patch(
                    all_perimeters=all_perimeters,
                    date_str=date_str,
                    next_date_str=(date + pd.Timedelta("1D")).strftime("%Y-%m-%d"),
                    transform=transform,
                    shape=shape,
                    year=year,
                )
            except Exception as exc:
                log.warning(f"Skipping {fire_id} {date_str}: {exc}")
                continue

            n_burn = int(target_array.sum())
            n_total = int(target_array.size)

            patch_id = f"{fire_id}_{date_str}"
            out_path = self.output_dir / f"{patch_id}.npz"
            np.savez_compressed(
                out_path,
                X=patch_array,          # (18, 64, 64) float32
                y=target_array,         # (64, 64) float32 — next-day burn mask
                transform=np.array(transform),
                crs=OUTPUT_CRS.to_wkt(),
            )

            # Convert centroid back to WGS84 for metadata
            import pyproj
            transformer = pyproj.Transformer.from_crs(
                OUTPUT_CRS.to_epsg(), 4326, always_xy=True
            )
            lon, lat = transformer.transform(cx, cy)

            yield PatchMeta(
                patch_id=patch_id,
                fire_id=fire_id,
                date=date_str,
                year=year,
                province=self.province,
                centre_lon=round(lon, 5),
                centre_lat=round(lat, 5),
                centre_x=round(cx, 1),
                centre_y=round(cy, 1),
                n_burn_pixels=n_burn,
                n_total_pixels=n_total,
                burn_fraction=round(n_burn / max(n_total, 1), 4),
            )

    def _build_patch(
        self,
        all_perimeters: gpd.GeoDataFrame,
        date_str: str,
        next_date_str: str,
        transform: rasterio.transform.Affine,
        shape: tuple[int, int],
        year: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the (18, H, W) input array and (H, W) target array for one patch.

        Raises RuntimeError if critical files are missing.
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
        era5_path = self._find_era5(date_str)
        if era5_path:
            era5_data = self._load_era5_patch(era5_path, transform, shape)
            for var_name in ["wind_dir", "wind_speed", "temp_min", "temp_max", "humidity", "precip"]:
                if var_name in era5_data:
                    X[ch[var_name]] = era5_data[var_name]

        # --- FWI system ---
        for fwi_var in ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]:
            fwi_path = self._find_fwi(date_str, fwi_var)
            if fwi_path and fwi_path.exists():
                X[ch[fwi_var]] = reproject_to_patch_grid(
                    fwi_path, transform, OUTPUT_CRS, shape
                )

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
                resampling=Resampling.nearest,
            )

        # --- Previous fire mask (t=0, input) ---
        prev_mask = burn_mask_from_perimeters(
            all_perimeters, date_str, transform, shape, OUTPUT_CRS
        )
        X[ch["PrevFireMask"]] = prev_mask

        # --- Target: next-day fire mask (t+1) ---
        next_mask = burn_mask_from_perimeters(
            all_perimeters, next_date_str, transform, shape, OUTPUT_CRS
        )

        return X, next_mask

    # ------------------------------------------------------------------
    # File finders
    # ------------------------------------------------------------------

    def _find_era5(self, date_str: str) -> Optional[Path]:
        """Return path to ERA5 NetCDF for the given date."""
        year, month = date_str[:4], date_str[5:7]
        candidates = [
            self.era5_dir / f"era5_{year}{month}.nc",
            self.era5_dir / f"era5_{year}.nc",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _find_fwi(self, date_str: str, variable: str) -> Optional[Path]:
        """Return path to CWFIS FWI GeoTIFF for the given date and variable."""
        date_nodash = date_str.replace("-", "")
        candidates = [
            self.fwi_dir / f"{variable}_{date_nodash}.tif",
            self.fwi_dir / date_str[:4] / f"{variable}_{date_nodash}.tif",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

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

    def _load_era5_patch(
        self,
        era5_path: Path,
        transform: rasterio.transform.Affine,
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        """
        Load ERA5 variables from a NetCDF file, compute derived fields,
        and reproject to patch grid.

        ERA5 variables used:
          u10, v10  → wind_dir (degrees), wind_speed (m/s)
          t2m       → temp_min, temp_max (°C) — uses daily min/max if available
          d2m       → dewpoint → relative humidity (%)
          tp        → total precipitation (mm)

        Returns dict mapping channel name → (H, W) float32 array.
        """
        import xarray as xr

        ds = xr.open_dataset(era5_path)
        result: dict[str, np.ndarray] = {}

        # Wind — derive speed and direction from u/v components
        if "u10" in ds and "v10" in ds:
            u = ds["u10"].values.squeeze().astype(np.float32)
            v = ds["v10"].values.squeeze().astype(np.float32)
            speed = np.sqrt(u**2 + v**2)
            direction = (np.degrees(np.arctan2(-u, -v)) % 360).astype(np.float32)
            result["wind_speed"] = self._regrid_numpy(speed, ds, transform, shape)
            result["wind_dir"] = self._regrid_numpy(direction, ds, transform, shape)

        # Temperature
        if "t2m" in ds:
            t_k = ds["t2m"].values.squeeze().astype(np.float32)
            t_c = t_k - 273.15
            result["temp_max"] = self._regrid_numpy(t_c, ds, transform, shape)
            result["temp_min"] = self._regrid_numpy(t_c, ds, transform, shape)  # daily if single step

        # Relative humidity from dewpoint
        if "d2m" in ds and "t2m" in ds:
            t = ds["t2m"].values.squeeze().astype(np.float32) - 273.15
            td = ds["d2m"].values.squeeze().astype(np.float32) - 273.15
            rh = 100.0 * np.exp((17.625 * td) / (243.04 + td)) / np.exp(
                (17.625 * t) / (243.04 + t)
            )
            result["humidity"] = self._regrid_numpy(rh.astype(np.float32), ds, transform, shape)

        # Precipitation (m → mm)
        if "tp" in ds:
            tp = ds["tp"].values.squeeze().astype(np.float32) * 1000.0
            result["precip"] = self._regrid_numpy(tp, ds, transform, shape)

        ds.close()
        return result

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
        lats = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
        lons = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values

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

    stats = {
        ch: {"mean": float(mean[i]), "std": float(std[i])}
        for i, ch in enumerate(CHANNELS)
    }
    out = stats_dir / "channel_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Channel stats saved → {out}")
