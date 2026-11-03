"""The live sources' response formats, faked from real samples, and the spread script's helpers."""

import io
import json
import re

import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.warp
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds, from_origin

from api.services import bc_fires
from scripts.predict_spread import BANDS, known_fires, spread_bands
from src.data import live
from src.data.assemble import OUTPUT_CRS, PATCH_EXTENT_M
from src.utils.config import TARGET_RESOLUTION_M

SP_HEADER = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,"
    "confidence,version,bright_ti5,frp,daynight,type"
)
NRT_HEADER = SP_HEADER.removesuffix(",type")


def tif_bytes(value: float) -> bytes:
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=2, height=2, count=1, dtype="float32", crs="EPSG:4326",
                      transform=from_origin(-140, 60, 0.25, 0.25)) as dst:
            dst.write(np.full((1, 2, 2), value, dtype=np.float32))
        return mem.read()


@pytest.mark.parametrize("tp30, precip", [(0.012, 10.0), (0.001, 0.0)])
def test_ecmwf_day_turns_three_hourly_fields_into_daily_ones(monkeypatch, tmp_path, tp30, precip):
    def values(step):
        return {"2t": 293.15, "2d": 283.15, "10u": step, "10v": -step, "mx2t3": 273.15 + step,
                "mn2t3": 273.15 + step, "tp": tp30 if step == 30 else 0.002}

    blobs, index = {}, {}
    for step in range(6, 31, 3):
        url, blob, lines = live.ECMWF_URL.format(d="20260801", step=step), b"", []
        for param, value in {"msl": 1e5, **values(step)}.items():
            data = tif_bytes(value)
            lines.append(json.dumps({"step": str(step), "param": param, "_offset": len(blob), "_length": len(data)}))
            blob += data
        blobs[url + ".grib2"], index[url + ".index"] = blob, "\n".join(lines).encode()

    def urlopen(req, timeout):
        first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", req.get_header("Range")).groups())
        return io.BytesIO(blobs[req.full_url][first:last + 1])

    monkeypatch.setattr(live, "_get", lambda url: index[url])
    monkeypatch.setattr(live.urllib.request, "urlopen", urlopen)
    day = live.ecmwf_day("2026-08-01", tmp_path)

    assert day["temp_max"].values[0, 0] == pytest.approx(30)
    assert day["temp_min"].values[0, 0] == pytest.approx(9)  # step 6 ends before the day starts
    assert day["humidity"].values[0, 0] == pytest.approx(52.6, abs=0.5)
    assert day["u10"].values[0, 0] == pytest.approx(22.5)  # 16:00 PDT, between steps 21 and 24
    assert day["v10"].values[0, 0] == pytest.approx(-22.5)
    assert day["precip"].values[0, 0] == pytest.approx(precip, abs=1e-3)
    assert day["precip"].latitude.values[0] == pytest.approx(59.875)  # cell centres, as in ERA5
    assert day["precip"].longitude.values[0] == pytest.approx(-139.875)


def firms_row(date, confidence="n"):
    return f"53.2,-125.0,346,0.4,0.6,{date},853,N20,VIIRS,{confidence},2,281,6,N"


def fake_firms(monkeypatch, area):
    calls = []
    availability = "data_id,min_date,max_date\n" + "".join(
        f"{sat}_SP,2018-04-01,2026-07-31\n{sat}_NRT,2026-08-01,2026-08-05\n" for sat in live.FIRMS_SATELLITES
    )

    def get(url):
        if "data_availability" in url:
            return availability.encode()
        calls.append(url)
        return area(url).encode()

    monkeypatch.setattr(live, "_get", get)
    return calls


def test_firms_uses_the_archive_where_it_reaches_and_caches_only_that(monkeypatch, tmp_path):
    def area(url):
        if "_NRT/" in url:  # near-real-time rows have no type column
            return f"{NRT_HEADER}\n{firms_row('2026-08-02')}\n{firms_row('2026-08-02', 'l')}\n"
        return (f"{SP_HEADER}\n{firms_row('2026-07-26')},0\n{firms_row('2026-07-26', 'l')},0\n"
                f"{firms_row('2026-07-26')},2\n")

    calls = fake_firms(monkeypatch, area)
    start, end = pd.Timestamp("2026-07-25"), pd.Timestamp("2026-08-05")
    df = live.firms_detections(start, end, "KEY", tmp_path)

    requested = [re.search(r"/(VIIRS_\w+)/.*?/(\d)/(\S+)$", u).groups() for u in calls]
    assert requested == [
        ("VIIRS_NOAA20_SP", "5", "2026-07-25"), ("VIIRS_NOAA21_SP", "5", "2026-07-25"),
        ("VIIRS_NOAA20_NRT", "5", "2026-07-30"), ("VIIRS_NOAA21_NRT", "5", "2026-07-30"),
        ("VIIRS_NOAA20_NRT", "2", "2026-08-04"), ("VIIRS_NOAA21_NRT", "2", "2026-08-04"),
    ]
    # One kept row per response: low confidence and non-vegetation rows are dropped
    assert len(df) == 6
    assert list(df.columns) == ["x", "y", "date"]
    x, y = rasterio.warp.transform("EPSG:4326", OUTPUT_CRS, [-125.0], [53.2])
    assert df["x"].iloc[0] == pytest.approx(x[0])
    assert df["y"].iloc[0] == pytest.approx(y[0])

    calls.clear()
    live.firms_detections(start, end, "KEY", tmp_path)
    assert not any("_SP/" in u for u in calls)
    assert len(calls) == 4


def test_firms_reports_an_error_body_instead_of_parsing_it(monkeypatch, tmp_path):
    fake_firms(monkeypatch, lambda url: "Invalid MAP_KEY.")
    with pytest.raises(RuntimeError, match="Invalid MAP_KEY"):
        live.firms_detections(pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-02"), "KEY", tmp_path)


def assembler(tmp_path):
    a = live.LiveAssembler.__new__(live.LiveAssembler)
    a.cache_dir, a.ndvi_dir = tmp_path, tmp_path
    return a


def test_cwfis_nodata_is_filled_from_its_neighbours(monkeypatch, tmp_path):
    centre_x, centre_y = rasterio.warp.transform(OUTPUT_CRS, "EPSG:3978", [1_200_000], [1_000_000])
    transform = from_origin(centre_x[0] - 100_000, centre_y[0] + 100_000, 2000, 2000)
    grid = np.full((100, 100), 20, dtype=np.float32)
    grid[48:52, 48:52] = -3.4e38  # a hole in the middle of the patch
    for var in live.FWI_VARS:
        with rasterio.open(tmp_path / f"cwfis_{var.lower()}_2026-08-01.tif", "w", driver="GTiff", width=100, height=100,
                           count=1, dtype="float32", crs="EPSG:3978", transform=transform, nodata=-3.4e38) as dst:
            dst.write(grid, 1)

    half = PATCH_EXTENT_M / 4
    patch = from_bounds(1_200_000 - half, 1_000_000 - half, 1_200_000 + half, 1_000_000 + half, 32, 32)
    out = assembler(tmp_path)._load_fwi_patch("2026-08-01", patch, (32, 32))
    assert set(out) == set(live.FWI_VARS)
    assert np.allclose(out["ISI"], 20)


def test_cwfis_error_page_is_not_cached_as_a_grid(monkeypatch, tmp_path):
    monkeypatch.setattr(live, "_get", lambda url: b"<ServiceExceptionReport>no such time</ServiceExceptionReport>")
    with pytest.raises(RuntimeError, match="ServiceException"):
        live.cwfis_grid("FWI", "2026-08-01", tmp_path)
    assert not list(tmp_path.iterdir())


def test_ndvi_falls_back_to_the_composite_closest_in_season(tmp_path):
    for name in ("ndvi_20250601", "ndvi_20250712", "ndvi_20250725"):
        (tmp_path / f"{name}.tif").touch()
    assert assembler(tmp_path)._find_ndvi("2026-07-20").name == "ndvi_20250712.tif"


def test_this_seasons_ndvi_wins_over_the_fallback(tmp_path):
    for name in ("ndvi_20250712", "ndvi_20260630"):
        (tmp_path / f"{name}.tif").touch()
    assert assembler(tmp_path)._find_ndvi("2026-07-20").name == "ndvi_20260630.tif"


def test_spread_bands_cut_at_the_thresholds_and_leave_out_burned_and_distant_land():
    size = 64
    highest, elevated = BANDS[0][1], BANDS[1][1]
    prob = np.full((size, size), elevated)
    prob[:, :5] = highest
    prob[:, 5:10] = elevated - 0.01
    burned = np.zeros((size, size))
    burned[32, 0] = 1
    band = spread_bands(prob, burned)

    assert band[32, 1] == 1
    assert band[32, 7] == 0
    assert band[32, 12] == 2
    assert band[32, 0] == 0  # already burned
    reach_px = int(15_000 / TARGET_RESOLUTION_M)
    assert band[32, reach_px] == 2
    assert band[32, reach_px + 1] == 0


def fire_feature(number, lon, lat):
    return {"properties": {"fire_number": number, "fire_name": number, "ignition_date": "2026-08-01"},
            "geometry": {"type": "Point", "coordinates": [lon, lat]}}


def fake_feeds(monkeypatch, active, perimeters):
    async def get_active_fires():
        return active

    async def get_historical_perimeters(year):
        return perimeters

    monkeypatch.setattr(bc_fires, "get_active_fires", get_active_fires)
    monkeypatch.setattr(bc_fires, "get_historical_perimeters", get_historical_perimeters)


def test_known_fires_prefers_the_mapped_perimeter(monkeypatch):
    ring = [[-125.1, 53.1], [-124.9, 53.1], [-124.9, 53.3], [-125.1, 53.1]]
    perimeter = {"properties": {"fire_number": "G1"}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
    fake_feeds(monkeypatch, {"features": [fire_feature("G1", -125, 53.2), fire_feature("G2", -120, 50)]},
               {"features": [perimeter]})
    fires = known_fires(2026).set_index("fire_number")
    assert fires.crs == OUTPUT_CRS
    assert fires.geometry["G1"].geom_type == "Polygon"
    assert fires.geometry["G2"].geom_type == "Point"


def test_known_fires_copes_with_no_fires_listed(monkeypatch):
    fake_feeds(monkeypatch, {"features": []}, {"features": []})
    assert known_fires(2026).empty


def test_known_fires_fails_when_the_feed_is_down(monkeypatch):
    fake_feeds(monkeypatch, None, None)
    with pytest.raises(RuntimeError, match="active fires"):
        known_fires(2026)
