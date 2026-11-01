"""
Tests for src/data/dataset.py and src/data/transforms.py

Uses synthetic .npz patches and manifests so no real data is required.
Verifies shapes, dtypes, NaN handling, normalisation, and augmentations.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.config import N_CHANNELS, PATCH_SIZE


# ---------------------------------------------------------------------------
# WildfireDataset tests
# ---------------------------------------------------------------------------

class TestWildfireDataset:
    def test_len(self, synthetic_dataset):
        from src.data.dataset import WildfireDataset
        ds = synthetic_dataset
        dataset = WildfireDataset(
            manifest_path=ds["splits_dir"] / "train.json",
            patches_dir=ds["patches_dir"],
            stats_path=ds["stats_dir"] / "channel_stats.json",
        )
        assert len(dataset) > 0

    def test_output_shapes(self, synthetic_dataset):
        from src.data.dataset import WildfireDataset
        ds = synthetic_dataset
        dataset = WildfireDataset(
            manifest_path=ds["splits_dir"] / "train.json",
            patches_dir=ds["patches_dir"],
            stats_path=ds["stats_dir"] / "channel_stats.json",
        )
        X, y = dataset[0]
        assert X.shape == (N_CHANNELS, PATCH_SIZE, PATCH_SIZE), \
            f"Expected X shape ({N_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}), got {X.shape}"
        assert y.shape == (PATCH_SIZE, PATCH_SIZE), \
            f"Expected y shape ({PATCH_SIZE}, {PATCH_SIZE}), got {y.shape}"

    def test_output_dtype(self, synthetic_dataset):
        from src.data.dataset import WildfireDataset
        ds = synthetic_dataset
        dataset = WildfireDataset(
            manifest_path=ds["splits_dir"] / "train.json",
            patches_dir=ds["patches_dir"],
            stats_path=ds["stats_dir"] / "channel_stats.json",
        )
        X, y = dataset[0]
        assert X.dtype == torch.float32, f"Expected float32, got {X.dtype}"
        assert y.dtype == torch.float32, f"Expected float32, got {y.dtype}"

    def test_no_nan_in_output(self, synthetic_dataset):
        from src.data.dataset import WildfireDataset
        ds = synthetic_dataset
        dataset = WildfireDataset(
            manifest_path=ds["splits_dir"] / "train.json",
            patches_dir=ds["patches_dir"],
            stats_path=ds["stats_dir"] / "channel_stats.json",
        )
        for i in range(min(5, len(dataset))):
            X, y = dataset[i]
            assert not torch.isnan(X).any(), f"NaN in X at index {i}"
            assert not torch.isnan(y).any(), f"NaN in y at index {i}"

    def test_target_is_binary(self, synthetic_dataset):
        from src.data.dataset import WildfireDataset
        ds = synthetic_dataset
        dataset = WildfireDataset(
            manifest_path=ds["splits_dir"] / "train.json",
            patches_dir=ds["patches_dir"],
            stats_path=ds["stats_dir"] / "channel_stats.json",
        )
        for i in range(min(5, len(dataset))):
            _, y = dataset[i]
            unique_vals = torch.unique(y)
            assert all(v in [0.0, 1.0] for v in unique_vals.tolist()), \
                f"Target contains non-binary values at index {i}: {unique_vals}"

    def test_normalisation_applied(self, synthetic_dataset):
        """After normalisation, values should not be in [0,1] raw range."""
        from src.data.dataset import WildfireDataset
        ds = synthetic_dataset
        dataset = WildfireDataset(
            manifest_path=ds["splits_dir"] / "train.json",
            patches_dir=ds["patches_dir"],
            stats_path=ds["stats_dir"] / "channel_stats.json",
        )
        X, _ = dataset[0]
        # With mean=0.5 and std=0.3, values (0–1) → centred around 0
        # The normalised X should have values outside [0, 1]
        assert X.min().item() < 0.0 or X.max().item() > 1.0, \
            "Normalisation does not seem to have been applied"

    def test_dataloader_batching(self, synthetic_dataset):
        from src.data.dataset import WildfireDataset
        from torch.utils.data import DataLoader
        ds = synthetic_dataset
        dataset = WildfireDataset(
            manifest_path=ds["splits_dir"] / "train.json",
            patches_dir=ds["patches_dir"],
            stats_path=ds["stats_dir"] / "channel_stats.json",
        )
        loader = DataLoader(dataset, batch_size=4)
        X_batch, y_batch = next(iter(loader))
        assert X_batch.shape[0] <= 4
        assert X_batch.shape[1] == N_CHANNELS
        assert X_batch.shape[2] == PATCH_SIZE
        assert y_batch.shape[1] == PATCH_SIZE


# ---------------------------------------------------------------------------
# WildfireSequenceDataset tests
# ---------------------------------------------------------------------------

class TestWildfireSequenceDataset:
    def test_output_shape_seq5(self, synthetic_dataset):
        from src.data.dataset import WildfireSequenceDataset
        ds = synthetic_dataset
        dataset = WildfireSequenceDataset(
            manifest_path=ds["splits_dir"] / "train.json",
            patches_dir=ds["patches_dir"],
            stats_path=ds["stats_dir"] / "channel_stats.json",
            seq_len=5,
        )
        if len(dataset) == 0:
            pytest.skip("Not enough consecutive patches for seq_len=5 with synthetic data")
        X_seq, y = dataset[0]
        assert X_seq.shape == (5, N_CHANNELS, PATCH_SIZE, PATCH_SIZE), \
            f"Expected (5, {N_CHANNELS}, {PATCH_SIZE}, {PATCH_SIZE}), got {X_seq.shape}"
        assert y.shape == (PATCH_SIZE, PATCH_SIZE)

    def test_seq_len_1_matches_single_frame(self, synthetic_dataset):
        from src.data.dataset import WildfireSequenceDataset, WildfireDataset
        ds = synthetic_dataset
        seq_ds = WildfireSequenceDataset(
            manifest_path=ds["splits_dir"] / "train.json",
            patches_dir=ds["patches_dir"],
            stats_path=ds["stats_dir"] / "channel_stats.json",
            seq_len=1,
        )
        if len(seq_ds) == 0:
            pytest.skip("Insufficient patches")
        X_seq, y = seq_ds[0]
        assert X_seq.shape[0] == 1   # T=1
        last_input = seq_ds.sequences[0][0][-1]
        npz = np.load(ds["patches_dir"] / f"{last_input['patch_id']}.npz")
        assert torch.equal(y, torch.from_numpy(npz["y"].astype(np.float32)))  # next day, not two


# ---------------------------------------------------------------------------
# Transforms tests
# ---------------------------------------------------------------------------

def _vector_channels() -> set[int]:
    from src.utils.config import CHANNELS, VECTOR_CHANNEL_PAIRS
    return {CHANNELS.index(c) for pair in VECTOR_CHANNEL_PAIRS for c in pair}


class TestTransforms:
    def _make_tensors(self):
        X = torch.rand(N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        y = (torch.rand(PATCH_SIZE, PATCH_SIZE) > 0.8).float()
        return X, y

    def test_hflip_changes_values(self):
        from src.data.transforms import RandomHorizontalFlip
        X, y = self._make_tensors()
        t = RandomHorizontalFlip(p=1.0)
        X_out, y_out = t(X, y)
        scalar = [i for i in range(N_CHANNELS) if i not in _vector_channels()]
        assert torch.allclose(X_out[scalar], torch.flip(X, dims=[-1])[scalar])
        assert torch.allclose(y_out, torch.flip(y, dims=[-1]))

    def test_vflip_changes_values(self):
        from src.data.transforms import RandomVerticalFlip
        X, y = self._make_tensors()
        t = RandomVerticalFlip(p=1.0)
        X_out, y_out = t(X, y)
        scalar = [i for i in range(N_CHANNELS) if i not in _vector_channels()]
        assert torch.allclose(X_out[scalar], torch.flip(X, dims=[-2])[scalar])

    def test_rot90_preserves_shape(self):
        from src.data.transforms import RandomRot90
        X, y = self._make_tensors()
        t = RandomRot90(p=1.0)
        X_out, y_out = t(X, y)
        assert X_out.shape == X.shape
        assert y_out.shape == y.shape

    @pytest.mark.parametrize("seed", range(8))
    def test_wind_still_points_from_fire_to_new_burn(self, seed):
        """After any flip/rotation, the wind vector must point from the fire to where it spread."""
        import random
        from src.data.transforms import build_train_transforms
        from src.utils.config import CHANNELS
        u, v = CHANNELS.index("wind_u"), CHANNELS.index("wind_v")
        X = torch.zeros(N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        y = torch.zeros(PATCH_SIZE, PATCH_SIZE)
        X[CHANNELS.index("PrevFireMask"), 32, 32] = 1.0
        y[31, 33] = 1.0                       # spread one pixel north-east (row 0 is north)
        X[u], X[v] = 1.0, 1.0                 # wind blowing toward the north-east

        random.seed(seed)
        X_out, y_out = build_train_transforms(1.0 if seed % 2 else 0.0, 1.0 if seed % 4 > 1 else 0.0, 1.0, 0.0)(X, y)
        fire = torch.nonzero(X_out[CHANNELS.index("PrevFireMask")])[0]
        burn = torch.nonzero(y_out)[0]
        d_row, d_col = (burn - fire).tolist()
        assert (X_out[u, 0, 0].item(), X_out[v, 0, 0].item()) == (d_col, -d_row)

    def test_noise_preserves_mask_channels(self):
        """Noise should NOT be applied to PrevFireMask."""
        from src.data.transforms import GaussianNoise
        X, y = self._make_tensors()
        X[17] = 0.5  # PrevFireMask
        t = GaussianNoise(std=1.0)  # large std to make effect obvious
        X_out, _ = t(X, y)
        assert torch.allclose(X_out[17], X[17]), "PrevFireMask should not be noised"

    def test_pipeline_no_op_eval(self):
        from src.data.transforms import build_eval_transforms
        X, y = self._make_tensors()
        t = build_eval_transforms()
        X_out, y_out = t(X, y)
        assert torch.allclose(X_out, X)
        assert torch.allclose(y_out, y)


# ---------------------------------------------------------------------------
# ChannelStats tests
# ---------------------------------------------------------------------------

class TestChannelStats:
    def test_missing_stats_file_falls_back_to_identity(self, tmp_path):
        from src.data.dataset import ChannelStats
        stats = ChannelStats(tmp_path / "nonexistent.json")
        X = np.ones((N_CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        X_norm = stats.normalize(X)
        np.testing.assert_allclose(X_norm, X, atol=1e-6)

    def test_normalisation_centres_data(self, tmp_path):
        from src.data.dataset import ChannelStats
        # Write stats with known mean=5.0, std=2.0 for all channels
        stats_dict = {ch: {"mean": 5.0, "std": 2.0} for ch in [
            "elevation", "slope_east", "slope_north", "wind_u", "wind_v",
            "temp_min", "temp_max", "humidity", "precip", "FFMC", "DMC",
            "DC", "ISI", "BUI", "FWI", "NDVI", "landcover", "PrevFireMask",
            "years_since_fire",
        ]}
        (tmp_path / "stats.json").write_text(json.dumps(stats_dict))
        stats = ChannelStats(tmp_path / "stats.json")
        X = np.full((N_CHANNELS, PATCH_SIZE, PATCH_SIZE), 5.0, dtype=np.float32)
        X_norm = stats.normalize(X)
        np.testing.assert_allclose(X_norm, 0.0, atol=1e-5)


def test_channel_stats_give_vector_pairs_zero_mean_and_one_scale(tmp_path):
    from src.data.assemble import compute_channel_stats
    from src.utils.config import CHANNELS
    X = np.ones((N_CHANNELS, 4, 4), dtype=np.float32)
    X[CHANNELS.index("wind_u")] = 3.0
    X[CHANNELS.index("wind_v")] = 4.0
    np.savez(tmp_path / "p.npz", X=X, y=np.zeros((4, 4), dtype=np.float32))
    (tmp_path / "train.json").write_text(json.dumps([{"patch_id": "p"}]))
    compute_channel_stats(tmp_path, tmp_path / "train.json", tmp_path)
    stats = json.loads((tmp_path / "channel_stats.json").read_text())
    assert stats["wind_u"] == stats["wind_v"] == {"mean": 0.0, "std": pytest.approx(np.sqrt(12.5))}
    assert stats["FWI"] == {"mean": 1.0, "std": 0.0}


def test_firms_masks_grow_from_detections_inside_the_perimeter():
    import pandas as pd
    from rasterio.transform import from_origin

    from src.data.assemble import firms_daily_masks

    transform = from_origin(0, 4000, 1000, 1000)  # 4 × 4 km, rows run south
    perimeter = np.zeros((4, 4), dtype=np.float32)
    perimeter[:, :2] = 1
    dates = list(pd.date_range("2023-07-01", periods=3))
    detections = pd.DataFrame({
        "x": [500, 1500, 1500, 3500, 500, 2500],
        "y": [3500, 3500, 2500, 3500, 500, 1500],
        "date": pd.to_datetime(
            ["2023-07-01", "2023-07-02", "2023-07-02", "2023-07-01", "2023-06-30", "2023-07-03"]
        ),
    })
    day0, day1, day2 = firms_daily_masks(detections, dates, transform, perimeter)
    # Burned before the first day counts as already burned on it
    assert day0.sum() == 2 and day0[0, 0] and day0[3, 0]
    assert day1.sum() == 4 and day1[0, 1] and day1[1, 1]
    # Growth past the mapped perimeter, connected to the fire inside it, counts
    assert day2.sum() == 5 and day2[2, 2]
    # An isolated detection outside every perimeter (slash pile) never counts
    assert not day2[0, 3]


def test_wide_fires_are_tiled_and_neighbouring_fires_count_as_burned(tmp_path):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    from src.data.assemble import OUTPUT_CRS, PATCH_EXTENT_M, PatchExtractor

    class Extractor(PatchExtractor):
        def _build_patch_channels(self, prev_mask, **_):
            return prev_mask[None]

    # A fire 1.5 patches wide and a small neighbour 5 km north of it, inside the same tile
    fire = box(0, 0, 1.5 * PATCH_EXTENT_M, 10_000)
    neighbour = box(10_000, 15_000, 12_000, 17_000)
    season = gpd.GeoDataFrame({"FIRE_ID": ["F", "N"]}, geometry=[fire, neighbour], crs=OUTPUT_CRS)
    fire_rows = season.iloc[[0]].assign(
        IGN_DATE=pd.Timestamp("2023-07-01"), OUT_DATE=pd.Timestamp("2023-07-03")
    )
    detections = pd.DataFrame({
        "x": [500, 90_500, 11_000],
        "y": [500, 500, 16_000],
        "date": pd.to_datetime(["2023-07-01", "2023-07-02", "2023-07-02"]),
    })
    metas = list(Extractor(tmp_path, tmp_path)._extract_fire_patches(season, detections, fire_rows, "F", 2023))
    assert sorted(m.patch_id for m in metas) == ["F_t0_2023-07-01", "F_t1_2023-07-01"]
    left = np.load(tmp_path / "F_t0_2023-07-01.npz")["y"]
    assert left.sum() == 2  # the fire's own detection and the neighbour's


def test_ndvi_uses_the_latest_composite_that_ended_by_the_patch_date(tmp_path):
    from types import SimpleNamespace

    from src.data.assemble import PatchExtractor

    for day in ("20210711", "20210727", "20210812"):
        (tmp_path / f"ndvi_{day}.tif").touch()
    find = lambda date: PatchExtractor._find_ndvi(SimpleNamespace(ndvi_dir=tmp_path), date)  # noqa: E731
    assert find("2021-07-27").name == "ndvi_20210727.tif"
    # The composite ending 08-12 covers 07-28 onward, so it could show the next day's burn
    assert find("2021-08-11").name == "ndvi_20210727.tif"
    assert find("2021-04-01") is None
    assert find("2021-10-01") is None


def test_years_since_fire_uses_the_latest_earlier_season_only():
    from types import SimpleNamespace

    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from src.data.assemble import OUTPUT_CRS, YEARS_SINCE_FIRE_CAP, PatchExtractor

    # 4 × 4 km patch; fires drawn on overlapping columns
    history = gpd.GeoDataFrame(
        {"FIRE_YEAR": [2000, 2010, 2015]},
        geometry=[box(0, 0, 3000, 4000), box(0, 0, 1000, 4000), box(0, 0, 4000, 4000)],
        crs=OUTPUT_CRS,
    )
    extractor = SimpleNamespace(fire_history=history)
    transform = from_origin(0, 4000, 1000, 1000)
    out = PatchExtractor._years_since_fire(extractor, 2015, transform, (4, 4))
    # The 2015 fire is this season's, so it must not count
    np.testing.assert_array_equal(out[0], [5, 15, 15, YEARS_SINCE_FIRE_CAP])
