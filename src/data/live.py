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
import io
import json
import logging
import urllib.request
from pathlib import Path

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


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def ecmwf_day(date_str: str, cache_dir: Path) -> dict[str, xr.DataArray]:
    """
    The same daily fields as PatchExtractor._era5_day, from that day's 00 UTC forecast.

    Steps 6–30 h cover 23:00–23:00 PDT, three-hourly; 16:00 PDT wind is the mean of 21 and 24 h.
    Only the needed fields are downloaded, by byte range from the run's index.
    """
    d = date_str.replace("-", "")
    steps = range(6, 31, 3)
    fields = {}
    for step in steps:
        url = ECMWF_URL.format(d=d, step=step)
        index = [json.loads(line) for line in _get(url + ".index").decode().splitlines()]
        for entry in index:
            if entry["param"] not in ("2t", "2d", "10u", "10v", "mx2t3", "mn2t3", "tp"):
                continue
            path = cache_dir / f"ecmwf_{d}_{step}_{entry['param']}.grib2"
            if not path.exists():
                req = urllib.request.Request(url + ".grib2", headers={
                    "Range": f"bytes={entry['_offset']}-{entry['_offset'] + entry['_length'] - 1}"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    path.write_bytes(resp.read())
            with rasterio.Env(GRIB_NORMALIZE_UNITS="NO"), rasterio.open(path) as src:
                fields[entry["param"], step] = src.read(1).astype(np.float32)
                transform, shape = src.transform, src.shape

    # Cell centres, as ERA5's coordinates are
    lats = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
    lons = transform.c + transform.a * (np.arange(shape[1]) + 0.5)

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
        path.write_bytes(_get(CWFIS_URL.format(var=var.lower(), d=date_str)))
    return path


def firms_detections(start: pd.Timestamp, end: pd.Timestamp, map_key: str, cache_dir: Path) -> pd.DataFrame:
    """
    VIIRS detections over BC from start to end (inclusive) as patch-CRS points, filtered as in
    training. The API serves 5 days per request: from the standard archive while it reaches,
    near-real-time after. Archive chunks are cached.
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
    df = pd.concat(frames, ignore_index=True)
    # Near-real-time rows have no type (vegetation fire or not); the perimeter filter drops other burning
    df = df[(df["confidence"] != "l") & (df.get("type", pd.Series(0, index=df.index)).fillna(0) == 0)]
    x, y = rasterio.warp.transform("EPSG:4326", OUTPUT_CRS, df["longitude"].to_numpy(), df["latitude"].to_numpy())
    return pd.DataFrame({"x": x, "y": y, "date": pd.to_datetime(df["acq_date"]).to_numpy()})


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
