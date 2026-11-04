"""
Live inputs for the spread model: the training channels, from sources available on the day.

  - weather       ECMWF open-data forecast (the model family behind ERA5), 0.25°
  - fire weather  CWFIS daily grids, 2 km (scoring 2024 with these in place of CEMS cost nothing)
  - fire mask     FIRMS VIIRS near-real-time detections (NOAA-20 and NOAA-21)

Terrain, land cover, vegetation and fire history come from the same raw files as
training, through PatchExtractor, so live patches are built exactly like training ones.
"""

from __future__ import annotations

import functools
import http.client
import io
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import xarray as xr

from src.data.assemble import OUTPUT_CRS, PatchExtractor, _fill_nan, reproject_to_patch_grid

log = logging.getLogger(__name__)

ECMWF_URL = "https://data.ecmwf.int/forecasts/{d}/00z/ifs/0p25/oper/{d}000000-{step}h-oper-fc"
CWFIS_URL = (
    "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/wcs?service=WCS&version=2.0.1&request=GetCoverage"
    "&coverageId=public__{var}&format=image/geotiff&subset=time(%22{d}T00:00:00.000Z%22)"
    "&subset=E(-2400000,-1000000)&subset=N(400000,2200000)"  # BC, in CWFIS's EPSG:3978
)
FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api"
# Suomi NPP retired on 1 Nov 2026; its two successors give the same two-VIIRS coverage as training
FIRMS_SATELLITES = ["VIIRS_NOAA20", "VIIRS_NOAA21"]
FWI_VARS = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
ECMWF_PARAMS = ("2t", "2d", "10u", "10v", "mx2t3", "mn2t3", "tp")
ECMWF_ARCHIVE_STEPS = range(6, 55, 3)  # the day T of the forecast run, then the day being predicted
BC_LATITUDES = slice(61, 47)  # the archive's crop, north to south as in the forecast grid
BC_LONGITUDES = slice(-141, -113)
GET_ATTEMPTS = 3
RETRY_SECONDS = 30
CACHE_KEEP_DAYS = 7  # archived days stay in the archive; these are the full-size downloads


def _get(url: str, headers: dict | None = None) -> bytes:
    """The response body, retrying what may pass: dropped connections and server errors, not a 404."""
    for attempt in range(GET_ATTEMPTS):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120) as resp:
                return resp.read()
        except (OSError, http.client.HTTPException) as exc:
            client_error = isinstance(exc, urllib.error.HTTPError) and exc.code < 500
            if client_error or attempt == GET_ATTEMPTS - 1:
                raise
            time.sleep(RETRY_SECONDS * (attempt + 1))


def _ecmwf_fields(date_str: str, steps: range, cache_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Each (param, step) of the day's 00 UTC forecast, by byte range from its index, and the grid's cell centres."""
    d = date_str.replace("-", "")
    fields = {}
    for step in steps:
        url = ECMWF_URL.format(d=d, step=step)
        index = [json.loads(line) for line in _get(url + ".index").decode().splitlines()]
        for entry in index:
            if entry["param"] not in ECMWF_PARAMS:
                continue
            path = cache_dir / f"ecmwf_{d}_{step}_{entry['param']}.grib2"
            if not path.exists():
                last_byte = entry["_offset"] + entry["_length"] - 1
                path.write_bytes(_get(url + ".grib2", {"Range": f"bytes={entry['_offset']}-{last_byte}"}))
            with rasterio.Env(GRIB_NORMALIZE_UNITS="NO"), rasterio.open(path) as src:
                fields[entry["param"], step] = src.read(1).astype(np.float32)
                transform, shape = src.transform, src.shape

    # Cell centres, as ERA5's coordinates are
    lats = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
    lons = transform.c + transform.a * (np.arange(shape[1]) + 0.5)
    return fields, lats, lons


def ecmwf_day(date_str: str, cache_dir: Path) -> dict[str, xr.DataArray]:
    """
    The same daily fields as PatchExtractor._era5_day, from that day's 00 UTC forecast.

    Steps 6–30 h cover 23:00–23:00 PDT, three-hourly; 16:00 PDT wind is the mean of 21 and 24 h.
    """
    steps = range(6, 31, 3)
    fields, lats, lons = _ecmwf_fields(date_str, steps, cache_dir)

    def grid(a: np.ndarray) -> xr.DataArray:
        return xr.DataArray(a, coords={"latitude": lats, "longitude": lons}, dims=("latitude", "longitude"))

    hours = steps[1:]  # 02:00–23:00 PDT, and the 3 h max/min ending at each
    t = np.stack([fields["2t", s] for s in hours]) - 273.15
    td = np.stack([fields["2d", s] for s in hours]) - 273.15
    rh = 100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))
    return {
        "temp_max": grid(np.max([fields["mx2t3", s] for s in hours], axis=0) - 273.15),
        "temp_min": grid(np.min([fields["mn2t3", s] for s in hours], axis=0) - 273.15),
        "humidity": grid(rh.min(axis=0)),
        "u10": grid((fields["10u", 21] + fields["10u", 24]) / 2),
        "v10": grid((fields["10v", 21] + fields["10v", 24]) / 2),
        "precip": grid(np.maximum(fields["tp", 30] - fields["tp", 6], 0) * 1000.0),  # m → mm; packing can dip below 0
    }


def cwfis_grid(var: str, date_str: str, cache_dir: Path) -> Path:
    """CWFIS's daily grid of one FWI-system index over BC, downloaded once."""
    path = cache_dir / f"cwfis_{var.lower()}_{date_str}.tif"
    if not path.exists():
        data = _get(CWFIS_URL.format(var=var.lower(), d=date_str))
        if not data.startswith((b"II", b"MM")):  # a GeoTIFF; a cached error page would break every later run
            raise RuntimeError(f"CWFIS {var} {date_str}: {data[:200]!r}")
        path.write_bytes(data)
    return path


def _firms_rows(start: pd.Timestamp, end: pd.Timestamp, map_key: str, cache_dir: Path) -> pd.DataFrame:
    """
    FIRMS's VIIRS rows over BC from start to end (inclusive). The API serves 5 days per request:
    from the standard archive while it reaches, near-real-time after. Archive chunks are cached.
    """
    avail = pd.read_csv(io.StringIO(_get(f"{FIRMS_API}/data_availability/csv/{map_key}/all").decode()),
                        index_col=0, parse_dates=["max_date"])
    frames = []
    for chunk in pd.date_range(start, end, freq="5D"):
        days = min(5, (end - chunk).days + 1)
        for sat in FIRMS_SATELLITES:
            archived = chunk + pd.Timedelta(days=days - 1) <= avail.loc[f"{sat}_SP", "max_date"]
            source = f"{sat}_SP" if archived else f"{sat}_NRT"
            path = cache_dir / f"firms_{source}_{chunk:%Y-%m-%d}_{days}.csv"
            if path.exists():
                text = path.read_text()
            else:
                text = _get(f"{FIRMS_API}/area/csv/{map_key}/{source}/-139,48,-114,60/{days}/{chunk:%Y-%m-%d}").decode()
                if not text.startswith("latitude"):
                    raise RuntimeError(f"FIRMS {source} {chunk:%Y-%m-%d}: {text[:200]}")
                if archived:
                    path.write_text(text)
            frames.append(pd.read_csv(io.StringIO(text)))
    return pd.concat(frames, ignore_index=True)


def firms_detections(start: pd.Timestamp, end: pd.Timestamp, map_key: str, cache_dir: Path) -> pd.DataFrame:
    """VIIRS detections over BC from start to end (inclusive) as patch-CRS points, filtered as in training."""
    df = _firms_rows(start, end, map_key, cache_dir)
    # Near-real-time rows have no type (vegetation fire or not); the perimeter filter drops other burning
    df = df[(df["confidence"] != "l") & (df.get("type", pd.Series(0, index=df.index)).fillna(0) == 0)]
    x, y = rasterio.warp.transform("EPSG:4326", OUTPUT_CRS, df["longitude"].to_numpy(), df["latitude"].to_numpy())
    return pd.DataFrame({"x": x, "y": y, "date": pd.to_datetime(df["acq_date"]).to_numpy()})


def ecmwf_archive(date_str: str, cache_dir: Path) -> xr.Dataset:
    """The day's forecast fields cropped to BC, three-hourly and out to the day being predicted."""
    fields, lats, lons = _ecmwf_fields(date_str, ECMWF_ARCHIVE_STEPS, cache_dir)
    data = {p: (("step", "latitude", "longitude"), np.stack([fields[p, s] for s in ECMWF_ARCHIVE_STEPS]))
            for p in ECMWF_PARAMS}
    ds = xr.Dataset(data, coords={"step": list(ECMWF_ARCHIVE_STEPS), "latitude": lats, "longitude": lons})
    return ds.sel(latitude=BC_LATITUDES, longitude=BC_LONGITUDES)


def archive_inputs(day: pd.Timestamp, fires: gpd.GeoDataFrame, map_key: str, cache_dir: Path,
                   archive_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """
    Keep what a live run sees for day T, since ECMWF deletes its forecasts after about 4 days and
    near-real-time detections and the fire list are later replaced. Returns the files written and
    the sources that failed: those are logged and skipped, so they cannot cost the day's prediction.
    """
    date_str = f"{day:%Y-%m-%d}"
    out = archive_dir / date_str
    out.mkdir(parents=True, exist_ok=True)

    def write_ecmwf(path: Path) -> None:
        ds = ecmwf_archive(date_str, cache_dir)
        ds.to_netcdf(path, encoding={p: {"zlib": True} for p in ds.data_vars})

    def write_cwfis(var: str) -> Callable[[Path], None]:
        def write(path: Path) -> None:
            with rasterio.open(cwfis_grid(var, date_str, cache_dir)) as src:
                with rasterio.open(path, "w", **(src.profile | {"compress": "deflate", "predictor": 3})) as dst:
                    dst.write(src.read())
        return write

    def write_firms(path: Path) -> None:
        _firms_rows(day - pd.Timedelta(days=1), day, map_key, cache_dir).to_csv(path, index=False)

    sources = {
        "ecmwf.nc": write_ecmwf,
        **{f"cwfis_{var.lower()}.tif": write_cwfis(var) for var in FWI_VARS},
        "firms.csv.gz": write_firms,
        "fires.geojson": lambda path: path.write_text(fires.to_crs("EPSG:4326").to_json()),
    }
    saved, failed = {}, []
    for name, write in sources.items():
        tmp = out / f"tmp_{name}"  # a failed rewrite leaves the earlier file in place
        try:
            write(tmp)
            saved[name] = tmp.replace(out / name)
        except Exception:
            log.exception(f"Could not archive {name} for {date_str}")
            tmp.unlink(missing_ok=True)
            failed.append(name)
    return saved, failed


def prune_cache(cache_dir: Path) -> None:
    """Remove the full-size ECMWF and CWFIS downloads once they are too old to rerun a day from."""
    cutoff = time.time() - CACHE_KEEP_DAYS * 24 * 3600
    for path in [*cache_dir.glob("ecmwf_*"), *cache_dir.glob("cwfis_*")]:
        if path.stat().st_mtime < cutoff:
            path.unlink()


class LiveAssembler(PatchExtractor):
    """PatchExtractor with weather from ECMWF and fire weather from CWFIS."""

    def __init__(self, raw_dir: Path, cache_dir: Path):
        super().__init__(raw_dir, cache_dir)
        self.cache_dir = Path(cache_dir)

    @functools.lru_cache(maxsize=4)
    def _era5_day(self, date_str: str) -> dict:
        return ecmwf_day(date_str, self.cache_dir)

    def _load_fwi_patch(self, date_str, transform, shape) -> dict[str, np.ndarray]:
        out = {}
        for var in FWI_VARS:
            a = reproject_to_patch_grid(cwfis_grid(var, date_str, self.cache_dir), transform, OUTPUT_CRS, shape,
                                        fill_value=np.nan)
            out[var] = _fill_nan(np.where((a < 0) | (a > 5000), np.nan, a))  # nodata is -3.4e38
        return out

    def _find_ndvi(self, date_str: str) -> Path | None:
        """The prepared composite closest in season, from any year, when this season's is not ready."""
        if current := super()._find_ndvi(date_str):
            return current
        target = pd.Timestamp(date_str)
        # Days back to the composite's end, ignoring the year: 0–32 like training is best
        back = [((target - pd.Timestamp(f.stem[-8:])).days % 365, f) for f in self.ndvi_dir.glob("ndvi_*.tif")]
        return min(back)[1] if back else None
