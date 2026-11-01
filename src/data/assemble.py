"""
Canadian Wildfire Dataset Assembly Pipeline
============================================
Assembles spatially co-registered 64×64 patches from multiple open data sources:

  BC fire perimeters (SHP/GeoJSON)
  NASA FIRMS VIIRS active-fire detections (CSV)
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
import scipy.ndimage
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from src.utils.config import CHANNELS, PATCH_SIZE, TARGET_RESOLUTION_M, VECTOR_CHANNEL_PAIRS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Daily fire progression from satellite detections
# ---------------------------------------------------------------------------

def firms_daily_masks(
    detections: pd.DataFrame,          # x, y in the patch CRS; date (Timestamp)
    dates: list[pd.Timestamp],
    transform: rasterio.transform.Affine,
    perimeter: np.ndarray,             # (H, W) — rasterised final perimeters
) -> list[np.ndarray]:
    """
    Cumulative burn mask per day: a pixel has burned from the first day a detection
    falls in it. Only connected clusters of detections that reach into a recorded fire
    perimeter count: that keeps growth past a perimeter mapped before the fire stopped,
    and drops isolated burning that is not a wildfire, such as slash piles. Detections
    before the first day count as already burned, so pass one season's detections only.
    """
    rows, cols = rasterio.transform.rowcol(transform, detections["x"], detections["y"])
    rows, cols = np.asarray(rows), np.asarray(cols)
    day = (detections["date"] - dates[0]).dt.days.to_numpy()
    H, W = perimeter.shape
    day = np.maximum(day, 0)  # burned before the first day: already burned on it
    keep = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W) & (day < len(dates))
    first = np.full((H, W), len(dates))
    np.minimum.at(first, (rows[keep], cols[keep]), day[keep])
    burned = first < len(dates)
    clusters, _ = scipy.ndimage.label(burned, structure=np.ones((3, 3)))
    fire_clusters = np.unique(clusters[burned & (perimeter > 0)])
    first[~np.isin(clusters, fire_clusters[fire_clusters > 0])] = len(dates)
    return [(first <= i).astype(np.float32) for i in range(len(dates))]

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

# NFDB perimeters start in 1972, 40 years before the first patch season (2012)
YEARS_SINCE_FIRE_CAP = 40

# Buffer around fire perimeter edge for patch sampling (metres)


# ---------------------------------------------------------------------------
# Patch metadata record
# ---------------------------------------------------------------------------

@dataclass
class PatchMeta:
    patch_id: str          # "{fire_id}_{date}", fire_id with _t{k} for tiled fires
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


# ---------------------------------------------------------------------------
# Terrain derivation
# ---------------------------------------------------------------------------

def _fill_nan(a: np.ndarray) -> np.ndarray:
    """Fill pixels with no source data with the mean of the rest (0 if there is none)."""
    return np.where(np.isnan(a), np.nanmean(a) if not np.isnan(a).all() else 0.0, a).astype(np.float32)


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
        self.firms_dir = self.raw_dir / "firms"
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
        self.fire_history = self._load_fire_history(perimeters)
        self.detections = self._load_detections()

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
          OUT_DATE  ← Oct 31 of the fire year (end of season)
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

        # OUT_DATE: season end. Days a fire did not grow in the VIIRS record are skipped later
        season_end = gdf["IGN_DATE"].dt.year.astype("Int64").astype(str) + "-10-31"
        gdf["OUT_DATE"] = pd.to_datetime(season_end, errors="coerce")

        # Infer FIRE_YEAR from IGN_DATE if not set
        mask_no_year = gdf["FIRE_YEAR"].isna()
        gdf.loc[mask_no_year, "FIRE_YEAR"] = gdf.loc[mask_no_year, "IGN_DATE"].dt.year

        # Drop fires with no geometry or no ignition date
        gdf = gdf[gdf.geometry.notna() & gdf["IGN_DATE"].notna()].copy()
        gdf["FIRE_YEAR"] = gdf["FIRE_YEAR"].astype(int)

        log.info(f"Loaded {len(gdf):,} fire perimeters ({gdf['FIRE_YEAR'].min()}–{gdf['FIRE_YEAR'].max()})")
        return gdf

    def _load_fire_history(self, perimeters: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """BC perimeters with their year in the patch CRS, oldest first: NFDB from 1972 + seasons."""
        parts = [perimeters[["FIRE_YEAR", "geometry"]].to_crs(OUTPUT_CRS)]
        for nfdb in sorted(self.perimeters_dir.glob("nfdb_extracted/NFDB_poly_*.shp")):
            # The filter column must be read too, or the filter silently matches nothing
            old = gpd.read_file(
                nfdb, where=f"SRC_AGENCY = '{self.province}'", columns=["YEAR", "SRC_AGENCY"]
            ).rename(columns={"YEAR": "FIRE_YEAR"})
            parts.append(old[["FIRE_YEAR", "geometry"]].to_crs(OUTPUT_CRS))
        history = pd.concat(parts, ignore_index=True)
        log.info(f"  Fire history: {len(history):,} perimeters from {history['FIRE_YEAR'].min()}")
        return history[history.geometry.notna()].sort_values("FIRE_YEAR")

    def _load_detections(self) -> pd.DataFrame:
        """VIIRS 375 m vegetation-fire detections (both satellites) as patch-CRS points."""
        files = sorted(self.firms_dir.glob("viirs-*_bc.csv"))
        if not files:
            raise FileNotFoundError(
                f"No VIIRS files in {self.firms_dir}. Run scripts/download_canada.py first."
            )
        cols = ["latitude", "longitude", "acq_date", "confidence", "type"]
        df = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
        df = df[(df["confidence"] != "l") & (df["type"] == 0)]
        # acq_date is UTC; BC's afternoon and night overpasses both fall on the local date
        lon, lat = df["longitude"].to_numpy(), df["latitude"].to_numpy()
        x, y = rasterio.warp.transform("EPSG:4326", OUTPUT_CRS, lon, lat)
        log.info(f"  VIIRS detections: {len(df):,} from {len(files)} files")
        return pd.DataFrame({"x": x, "y": y, "date": pd.to_datetime(df["acq_date"]).to_numpy()})

    def _years_since_fire(
        self, year: int, transform: rasterio.transform.Affine, shape: tuple[int, int]
    ) -> np.ndarray:
        """Years since each pixel last burned, from earlier seasons only so nothing leaks."""
        west, south, east, north = rasterio.transform.array_bounds(*shape, transform)
        history = self.fire_history
        earlier = history[history["FIRE_YEAR"] < year].cx[west:east, south:north]
        last = np.zeros(shape, dtype=np.float32)  # 0: no recorded fire
        if len(earlier):
            # Oldest first, so the most recent fire is drawn last and wins
            shapes = zip(earlier.geometry, earlier["FIRE_YEAR"].astype(float))
            rasterio.features.rasterize(shapes, out=last, transform=transform)
        since = np.minimum(year - last, YEARS_SINCE_FIRE_CAP)
        return np.where(last > 0, since, YEARS_SINCE_FIRE_CAP).astype(np.float32)

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

        # Every fire this season, any size: burning from a neighbouring fire is still burning
        season = perimeters[perimeters["FIRE_YEAR"] == year].to_crs(OUTPUT_CRS)
        detections = self.detections[self.detections["date"].dt.year == year]
        for fire_id, fire_group in year_fires.groupby("FIRE_ID", dropna=False):
            fire_id_str = str(fire_id) if fire_id else f"unk_{year}"
            meta_list.extend(
                self._extract_fire_patches(season, detections, fire_group, fire_id_str, year)
            )

        return meta_list

    def _extract_fire_patches(
        self,
        season: gpd.GeoDataFrame,
        detections: pd.DataFrame,
        fire_geom: gpd.GeoDataFrame,
        fire_id: str,
        year: int,
    ) -> Iterator[PatchMeta]:
        """
        Yield patches for each day the fire grew. A fire wider than one patch is
        covered by a grid of adjacent patches, each with its own id ({fire_id}_t{k}).
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
        if len(date_range) < 2:
            return

        fire_albers = fire_geom.to_crs(OUTPUT_CRS)
        west, south, east, north = fire_albers.total_bounds
        nx = max(1, int(np.ceil((east - west) / PATCH_EXTENT_M)))
        ny = max(1, int(np.ceil((north - south) / PATCH_EXTENT_M)))
        # Grid centred on the fire's bounding box
        x0 = (west + east - nx * PATCH_EXTENT_M) / 2
        y0 = (south + north + ny * PATCH_EXTENT_M) / 2
        for k, (r, c) in enumerate(np.ndindex(ny, nx)):
            xmin, ymax = x0 + c * PATCH_EXTENT_M, y0 - r * PATCH_EXTENT_M
            tile_id = fire_id if nx * ny == 1 else f"{fire_id}_t{k}"
            yield from self._extract_tile_patches(
                season, detections, fire_albers, tile_id, year, date_range, xmin, ymax
            )

    def _extract_tile_patches(
        self,
        season: gpd.GeoDataFrame,
        detections: pd.DataFrame,
        fire_albers: gpd.GeoDataFrame,
        tile_id: str,
        year: int,
        date_range: list[pd.Timestamp],
        xmin: float,
        ymax: float,
    ) -> Iterator[PatchMeta]:
        """Patches for one 64 km tile of a fire, one per day of new burning."""
        import pyproj
        shape = (PATCH_SIZE, PATCH_SIZE)
        xmax, ymin = xmin + PATCH_EXTENT_M, ymax - PATCH_EXTENT_M
        transform = from_bounds(xmin, ymin, xmax, ymax, *shape)
        own = rasterio.features.rasterize(
            [(geom, 1.0) for geom in fire_albers.geometry if geom is not None],
            out_shape=shape, transform=transform, dtype=np.float32, fill=0.0,
        )
        if own.sum() == 0:
            return  # this part of the grid misses the fire

        # Detections count inside any of this season's perimeters, not just this fire's
        nearby = season.cx[xmin:xmax, ymin:ymax]
        burnable = rasterio.features.rasterize(
            [(geom, 1.0) for geom in nearby.geometry if geom is not None],
            out_shape=shape, transform=transform, dtype=np.float32, fill=0.0,
        )
        daily_masks = firms_daily_masks(detections, date_range, transform, np.maximum(own, burnable))

        cx, cy = xmin + PATCH_EXTENT_M / 2, ymax - PATCH_EXTENT_M / 2
        transformer = pyproj.Transformer.from_crs(OUTPUT_CRS.to_epsg(), 4326, always_xy=True)
        lon, lat = transformer.transform(cx, cy)

        # Yield one patch per consecutive day pair (T → T+1)
        for i, date in enumerate(date_range[:-1]):
            date_str = date.strftime("%Y-%m-%d")
            prev_mask = daily_masks[i]       # burn extent at day T (input)
            next_mask = daily_masks[i + 1]   # burn extent at day T+1 (target)

            # Skip uninformative pairs (no new burning)
            if not (next_mask > prev_mask).any():
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
                log.warning(f"Skipping {tile_id} {date_str}: {exc}")
                continue

            patch_id = f"{tile_id}_{date_str}"
            np.savez_compressed(
                self.output_dir / f"{patch_id}.npz",
                X=X,              # (N_CH, 64, 64)
                y=next_mask,      # (64, 64) — next-day burn mask (TARGET)
                transform=np.array(transform),
                crs=OUTPUT_CRS.to_wkt(),
            )

            yield PatchMeta(
                patch_id=patch_id, fire_id=tile_id, date=date_str, year=year,
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

        PrevFireMask is provided explicitly (from the FIRMS daily progression).
        All other channels are loaded from the downloaded data sources.
        """
        H, W = shape
        X = np.zeros((len(CHANNELS), H, W), dtype=np.float32)
        ch = {name: i for i, name in enumerate(CHANNELS)}

        # --- Terrain, land cover and fire history: the same every day at this location ---
        for var_name, arr in self._static_channels(transform, shape, year).items():
            X[ch[var_name]] = arr

        # --- ERA5 weather ---
        for var_name, arr in self._load_era5_patch(date_str, transform, shape).items():
            X[ch[var_name]] = arr

        # --- FWI system ---
        for var_name, arr in self._load_fwi_patch(date_str, transform, shape).items():
            X[ch[var_name]] = arr

        # --- NDVI (16-day composites from scripts/prepare_ndvi.py) ---
        ndvi_path = self._find_ndvi(date_str)
        if ndvi_path:
            ndvi = reproject_to_patch_grid(
                ndvi_path, transform, OUTPUT_CRS, shape, fill_value=np.nan
            )
            X[ch["NDVI"]] = _fill_nan(ndvi)

        # --- Previous fire mask (t=0) — provided by caller (FIRMS progression) ---
        X[ch["PrevFireMask"]] = prev_mask
        return X

    @functools.lru_cache(maxsize=16)
    def _static_channels(
        self, transform: rasterio.transform.Affine, shape: tuple[int, int], year: int
    ) -> dict[str, np.ndarray]:
        """Channels that do not change within a season. Cached: a tile yields a patch per day."""
        out = {}
        if self.cdem_path.exists():
            elev = reproject_to_patch_grid(self.cdem_path, transform, OUTPUT_CRS, shape, fill_value=np.nan)
            # SRTM voids and land outside BC take the patch's mean elevation
            elev = _fill_nan(np.where(elev <= -32767, np.nan, elev))
            # Gradient as east/north components, which flips and rotations transform exactly
            d_south, d_east = np.gradient(elev, TARGET_RESOLUTION_M)   # rows run south
            out.update(elevation=elev, slope_east=d_east, slope_north=-d_south)
        if self.landcover_path.exists():
            out["landcover"] = reproject_to_patch_grid(
                self.landcover_path,
                transform,
                OUTPUT_CRS,
                shape,
                resampling=Resampling.mode,  # dominant 30 m class in each 1 km pixel
            )
        # A recent burn has little fuel left to carry a fire
        out["years_since_fire"] = self._years_since_fire(year, transform, shape)
        return out

    # ------------------------------------------------------------------
    # File finders
    # ------------------------------------------------------------------

    def _find_ndvi(self, date_str: str) -> Optional[Path]:
        """
        The latest NDVI composite that ended by date_str, if it ended within 32 days.

        A composite still running after the patch date could already show the next day's burn.
        """
        target = pd.Timestamp(date_str)
        ended = [(pd.Timestamp(f.stem[-8:]), f) for f in self.ndvi_dir.glob("ndvi_*.tif")]
        ended = [(d, f) for d, f in ended if pd.Timedelta(0) <= target - d <= pd.Timedelta(days=32)]
        return max(ended)[1] if ended else None

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
        # Wind stays as components: no 360° → 0° wrap, and flips/rotations transform it exactly
        if "u10" in fields:
            fields["wind_u"], fields["wind_v"] = fields.pop("u10"), fields.pop("v10")
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
        """Regrid one field from its regular lat/lon grid to the patch grid, in memory."""
        lat_key = next((k for k in ["latitude", "lat"] if k in ds.coords), None)
        lon_key = next((k for k in ["longitude", "lon"] if k in ds.coords), None)
        if lat_key is None or lon_key is None:
            return np.full(shape, 0.0, dtype=np.float32)
        lats = ds.coords[lat_key].values
        lons = ds.coords[lon_key].values
        lon_res = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.25
        lat_res = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25

        # ERA5 and CEMS coordinates are cell centres; the transform needs the outer edge
        src_transform = rasterio.transform.from_origin(
            float(lons[0]) - lon_res / 2, float(lats[0]) + lat_res / 2, lon_res, lat_res
        )
        destination = np.full(shape, np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=(data if data.ndim == 2 else data[0]).astype(np.float32),
            destination=destination,
            src_transform=src_transform,
            src_crs=CRS.from_epsg(4326),
            dst_transform=transform,
            dst_crs=OUTPUT_CRS,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
        # CEMS is NaN over ocean, and patches can extend past the download box
        return _fill_nan(destination)


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

    Accumulates per-channel sums one patch at a time to avoid loading all patches into RAM.
    Vector channel pairs get mean 0 and one shared scale, so flips and rotations
    (sign changes and swaps of the components) stay valid after normalisation.
    """
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    with open(train_manifest) as f:
        train_records = json.load(f)

    n_channels = len(CHANNELS)
    count = 0
    total = np.zeros(n_channels, dtype=np.float64)
    total_sq = np.zeros(n_channels, dtype=np.float64)

    for record in train_records:
        npz_path = patches_dir / f"{record['patch_id']}.npz"
        if not npz_path.exists():
            continue
        X = np.load(npz_path)["X"].astype(np.float64).reshape(n_channels, -1)
        count += X.shape[1]
        total += X.sum(axis=1)
        total_sq += (X ** 2).sum(axis=1)

    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(total_sq / max(count, 1) - mean ** 2, 0.0))
    for pair in VECTOR_CHANNEL_PAIRS:
        idx = [CHANNELS.index(c) for c in pair]
        mean[idx] = 0.0
        std[idx] = np.sqrt(total_sq[idx].sum() / (2 * max(count, 1)))   # RMS over both components
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
