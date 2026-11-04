"""The live sources' response formats, faked from real samples, and the spread script's helpers."""

import http.client
import io
import json
import os
import re
import time
import urllib.error

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.warp
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds, from_origin
from shapely.geometry import Point

from api.services import bc_fires, spread_store
from scripts import predict_spread
from scripts.predict_spread import BANDS, known_fires, main, predict_fires, score_run, spread_bands, tile_score
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


def fake_ecmwf(monkeypatch, values, steps):
    blobs, index = {}, {}
    for step in steps:
        url, blob, lines = live.ECMWF_URL.format(d="20260801", step=step), b"", []
        for param, value in {"msl": 1e5, **values(step)}.items():
            data = tif_bytes(value)
            lines.append(json.dumps({"step": str(step), "param": param, "_offset": len(blob), "_length": len(data)}))
            blob += data
        blobs[url + ".grib2"], index[url + ".index"] = blob, "\n".join(lines).encode()

    def get(url, headers=None):
        if not headers:
            return index[url]
        first, last = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        return blobs[url][first:last + 1]

    monkeypatch.setattr(live, "_get", get)


@pytest.mark.parametrize("tp30, precip", [(0.012, 10.0), (0.001, 0.0)])
def test_ecmwf_day_turns_three_hourly_fields_into_daily_ones(monkeypatch, tmp_path, tp30, precip):
    def values(step):
        return {"2t": 293.15, "2d": 283.15, "10u": step, "10v": -step, "mx2t3": 273.15 + step,
                "mn2t3": 273.15 + step, "tp": tp30 if step == 30 else 0.002}

    fake_ecmwf(monkeypatch, values, range(6, 31, 3))
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


def ecmwf_values(step):
    return {"2t": 293.15, "2d": 283.15, "10u": step, "10v": -step, "mx2t3": 300, "mn2t3": 280, "tp": 0.001 * step}


def test_ecmwf_archive_keeps_every_step_out_to_the_day_predicted(monkeypatch, tmp_path):
    fake_ecmwf(monkeypatch, ecmwf_values, live.ECMWF_ARCHIVE_STEPS)
    ds = live.ecmwf_archive("2026-08-01", tmp_path)
    assert list(ds.step.values) == list(range(6, 55, 3))
    assert set(ds.data_vars) == set(live.ECMWF_PARAMS)
    assert ds["10u"].sel(step=54).values[0, 0] == pytest.approx(54)
    assert ds["tp"].sel(step=30).values[0, 0] == pytest.approx(0.03)


def test_ecmwf_archive_crops_to_bc():
    lats = np.arange(90, -90, -0.25)
    lons = np.arange(-180, 180, 0.25)
    ds = xr.Dataset(coords={"latitude": lats, "longitude": lons}).sel(
        latitude=live.BC_LATITUDES, longitude=live.BC_LONGITUDES)
    assert ds.latitude.max() == 61 and ds.latitude.min() == 47
    assert ds.longitude.min() == -141 and ds.longitude.max() == -113


def archive_fixtures(monkeypatch, tmp_path, fail=()):
    fake_ecmwf(monkeypatch, ecmwf_values, live.ECMWF_ARCHIVE_STEPS)
    real_get = live._get

    def get(url, headers=None):
        if "firms" in url:
            if "firms" in fail:
                raise OSError("FIRMS is down")
            if "data_availability" in url:
                return (b"data_id,min_date,max_date\n"
                        + b"".join(f"{s}_SP,2018-04-01,2026-07-31\n".encode() for s in live.FIRMS_SATELLITES))
            return f"{NRT_HEADER}\n{firms_row('2026-08-01')}\n".encode()
        return real_get(url, headers)

    monkeypatch.setattr(live, "_get", get)
    for var in live.FWI_VARS:
        (tmp_path / f"cwfis_{var.lower()}_2026-08-01.tif").write_bytes(tif_bytes(20))
    return gpd.GeoDataFrame({"fire_number": ["G1"]}, geometry=[Point(1_200_000, 1_000_000)], crs=OUTPUT_CRS)


def test_archive_inputs_keeps_each_source_readable(monkeypatch, tmp_path):
    fires = archive_fixtures(monkeypatch, tmp_path)
    saved, failed = live.archive_inputs(pd.Timestamp("2026-08-01"), fires, "KEY", tmp_path, tmp_path / "archive")

    assert failed == []
    assert set(saved) == {"ecmwf.nc", "firms.csv.gz", "fires.geojson",
                          *(f"cwfis_{v.lower()}.tif" for v in live.FWI_VARS)}
    assert saved["ecmwf.nc"].parent == tmp_path / "archive" / "2026-08-01"
    with xr.open_dataset(saved["ecmwf.nc"]) as ds:
        assert ds["10u"].sel(step=54).values[0, 0] == pytest.approx(54)
    with rasterio.open(saved["cwfis_isi.tif"]) as src:
        assert src.compression.name == "deflate"
        assert np.allclose(src.read(1), 20)
    assert list(pd.read_csv(saved["firms.csv.gz"])["acq_date"].unique()) == ["2026-08-01"]
    feature = json.loads(saved["fires.geojson"].read_text())["features"][0]
    assert feature["properties"]["fire_number"] == "G1"
    assert -140 < feature["geometry"]["coordinates"][0] < -113  # lon/lat


def test_a_failing_source_costs_only_itself_and_keeps_an_earlier_file(monkeypatch, tmp_path):
    fires = archive_fixtures(monkeypatch, tmp_path, fail=("firms",))
    out = tmp_path / "archive" / "2026-08-01"
    out.mkdir(parents=True)
    (out / "firms.csv.gz").write_bytes(b"from the first run")
    saved, failed = live.archive_inputs(pd.Timestamp("2026-08-01"), fires, "KEY", tmp_path, tmp_path / "archive")

    assert failed == ["firms.csv.gz"]
    assert "firms.csv.gz" not in saved
    assert "ecmwf.nc" in saved and "fires.geojson" in saved
    assert (out / "firms.csv.gz").read_bytes() == b"from the first run"
    assert not list(out.glob("tmp_*"))


def test_prune_cache_removes_only_old_downloads(tmp_path):
    old = time.time() - (live.CACHE_KEEP_DAYS + 1) * 24 * 3600
    for name, mtime in [("ecmwf_old.grib2", old), ("cwfis_old.tif", old), ("firms_old.csv", old),
                        ("ecmwf_new.grib2", time.time())]:
        (tmp_path / name).touch()
        os.utime(tmp_path / name, (mtime, mtime))
    live.prune_cache(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ecmwf_new.grib2", "firms_old.csv"]


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


def test_a_missing_stats_file_stops_the_run_and_says_so(monkeypatch, tmp_path):
    notes = []
    monkeypatch.setattr(predict_spread, "notify", notes.append)
    missing = tmp_path / "channel_stats.json"
    monkeypatch.setattr("sys.argv", ["predict_spread", "--checkpoint", "a.pt", "--stats", str(missing)])
    with pytest.raises(FileNotFoundError, match="channel_stats"):
        main()
    assert "FileNotFoundError" in notes[0]


def fires_and_buffers(*points):
    fires = gpd.GeoDataFrame({"fire_number": [f"G{i}" for i in range(len(points))], "fire_name": ""},
                             geometry=[Point(p) for p in points], crs=OUTPUT_CRS)
    return fires, fires.geometry.buffer(1000)


def test_a_failed_tile_costs_only_that_fire():
    fires, buffered = fires_and_buffers((1_000_000, 1_000_000), (1_500_000, 1_000_000))

    def predict_tile(fire, xmin, ymax):
        if fire.fire_number == "G0":
            raise ValueError("bad tile")
        return [{"properties": {"fire_number": fire.fire_number}}]

    features, failed = predict_fires(fires, buffered, predict_tile)
    assert failed == ["G0"]
    assert [f["properties"]["fire_number"] for f in features] == ["G1"]


def test_a_tile_not_burning_is_no_failure():
    fires, buffered = fires_and_buffers((1_000_000, 1_000_000))
    assert predict_fires(fires, buffered, lambda fire, xmin, ymax: None) == ([], [])


def test_a_tile_shared_by_two_fires_is_predicted_once():
    fires, buffered = fires_and_buffers((1_000_000, 1_000_000), (1_000_000, 1_000_000))
    calls = []
    predict_fires(fires, buffered, lambda fire, xmin, ymax: calls.append(fire.fire_number) or [])
    assert calls == ["G0"]


def test_every_burning_tile_failing_stops_the_run():
    fires, buffered = fires_and_buffers((1_000_000, 1_000_000), (1_500_000, 1_000_000))

    def predict_tile(fire, xmin, ymax):
        raise ValueError("no weather today")

    with pytest.raises(RuntimeError, match="G0"):
        predict_fires(fires, buffered, predict_tile)


def test_tile_score_counts_each_band_and_what_it_caught():
    band = np.array([[1, 1, 2], [2, 0, 0]])
    burned = np.array([[0, 0, 0], [0, 0, 1]])
    new_burn = np.array([[True, False, True], [False, True, False]])
    assert tile_score("G1", band, burned, new_burn) == {
        "fire_number": "G1", "unburned_km2": 5, "new_burn_km2": 3, "flagged_highest_km2": 2,
        "flagged_elevated_km2": 2, "hit_highest_km2": 1, "hit_elevated_km2": 1,
    }


def scored_fixture(monkeypatch, tmp_path, features):
    """One fire, burned on the day and growing 2 pixels east the next; pixels start at the fire. Returns the scores."""
    x, y = 1_200_500, 1_000_500
    gpd.GeoDataFrame({"fire_number": ["G1"], "fire_name": [""]}, geometry=[Point(x, y)], crs=OUTPUT_CRS).to_crs(
        "EPSG:4326").to_file(archive_path(tmp_path), driver="GeoJSON")
    saved = {}
    monkeypatch.setattr(spread_store, "run_features", lambda url, day: features(x, y))
    monkeypatch.setattr(spread_store, "save_scores", lambda url, day, scores: saved.update(day=day, scores=scores))
    detections = pd.DataFrame({
        "x": [x + 500, x + 1500, x + 2500, x + 9500],
        "y": [y + 500] * 4,
        "date": pd.to_datetime(["2026-08-01", "2026-08-02", "2026-08-02", "2026-08-03"]),
    })
    dates = list(pd.date_range("2026-07-30", "2026-08-03"))
    failed = score_run("url", "2026-08-01", detections, dates, tmp_path)
    assert failed == []
    return saved


def archive_path(tmp_path):
    path = tmp_path / "2026-08-01" / "fires.geojson"
    path.parent.mkdir(parents=True)
    return path


def area_feature(x, y, band):
    """The polygon of the 1 km pixel holding x, y, in lon/lat as stored."""
    xs, ys = [x - 400, x + 400, x + 400, x - 400, x - 400], [y - 400, y - 400, y + 400, y + 400, y - 400]
    lons, lats = rasterio.warp.transform(OUTPUT_CRS, "EPSG:4326", xs, ys)
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [list(map(list, zip(lons, lats)))]},
            "properties": {"band": band, "fire_number": "G1"}}


def test_score_run_counts_next_day_growth_inside_and_outside_the_bands(monkeypatch, tmp_path):
    saved = scored_fixture(monkeypatch, tmp_path, lambda x, y: [area_feature(x + 1500, y + 500, "highest"),
                                                                area_feature(x + 3500, y + 500, "elevated")])
    score = saved["scores"][0]
    assert saved["day"] == "2026-08-01"
    assert score["fire_number"] == "G1"
    assert score["new_burn_km2"] == 2  # later growth, on 3 Aug, is another day's
    assert score["flagged_highest_km2"] == 1
    assert score["flagged_elevated_km2"] == 1
    assert score["hit_highest_km2"] == 1
    assert score["hit_elevated_km2"] == 0


def test_score_run_counts_growth_on_a_day_with_nothing_flagged_as_missed(monkeypatch, tmp_path):
    saved = scored_fixture(monkeypatch, tmp_path, lambda x, y: [])
    score = saved["scores"][0]
    assert score["new_burn_km2"] == 2
    assert score["hit_highest_km2"] + score["hit_elevated_km2"] == 0


class FakeResponse(io.BytesIO):
    pass


@pytest.mark.parametrize("error", [
    urllib.error.URLError("connection reset"),
    urllib.error.HTTPError("u", 503, "busy", None, None),
    http.client.IncompleteRead(b"partial"),
])
def test_get_retries_what_may_pass(monkeypatch, error):
    outcomes = [error, FakeResponse(b"body")]

    def urlopen(req, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(live.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(live, "RETRY_SECONDS", 0)
    assert live._get("https://example.test") == b"body"


def test_get_gives_up_at_once_on_a_404_and_after_repeated_failures(monkeypatch):
    calls = []

    def fail_with(error):
        def urlopen(req, timeout):
            calls.append(req.full_url)
            raise error
        return urlopen

    monkeypatch.setattr(live, "RETRY_SECONDS", 0)
    monkeypatch.setattr(live.urllib.request, "urlopen", fail_with(urllib.error.HTTPError("u", 404, "gone", None, None)))
    with pytest.raises(urllib.error.HTTPError):
        live._get("https://example.test")
    assert len(calls) == 1

    calls.clear()
    monkeypatch.setattr(live.urllib.request, "urlopen", fail_with(urllib.error.URLError("down")))
    with pytest.raises(urllib.error.URLError):
        live._get("https://example.test")
    assert len(calls) == live.GET_ATTEMPTS


def test_get_sends_the_range_header(monkeypatch):
    seen = {}

    def urlopen(req, timeout):
        seen["range"] = req.get_header("Range")
        return FakeResponse(b"x")

    monkeypatch.setattr(live.urllib.request, "urlopen", urlopen)
    live._get("https://example.test", {"Range": "bytes=0-9"})
    assert seen["range"] == "bytes=0-9"
