"""
Tests for src/data/dataset.py and src/data/transforms.py

Uses synthetic .npz patches and manifests so no real data is required.
Verifies shapes, dtypes, NaN handling, normalisation, and augmentations.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.config import N_CHANNELS, PATCH_SIZE


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_synthetic_patch(
    patch_id: str,
    n_burn_pixels: int = 500,
    out_dir: Path = None,
) -> Path:
    """Write a synthetic .npz patch file and return its path."""
    rng = np.random.default_rng(hash(patch_id) % (2**32))

    X = rng.random((N_CHANNELS, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    y = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    burn_indices = rng.choice(PATCH_SIZE * PATCH_SIZE, size=n_burn_pixels, replace=False)
    y.ravel()[burn_indices] = 1.0

    out_path = (out_dir or Path(tempfile.gettempdir())) / f"{patch_id}.npz"
    np.savez_compressed(out_path, X=X, y=y)
    return out_path


def make_synthetic_manifest(
    patch_ids: list[str],
    n_burn_pixels: int = 500,
    base_year: int = 2020,
) -> list[dict]:
    """Return a list of manifest records for synthetic patches."""
    records = []
    for i, pid in enumerate(patch_ids):
        year = base_year + (i % 3)
        records.append({
            "patch_id": pid,
            "fire_id": f"fire_{i // 5}",
            "date": f"{year}-07-{(i % 28) + 1:02d}",
            "year": year,
            "province": "BC",
            "centre_lon": -125.0 + i * 0.1,
            "centre_lat": 51.0 + i * 0.05,
            "centre_x": 1000000.0,
            "centre_y": 600000.0,
            "n_burn_pixels": n_burn_pixels,
            "n_total_pixels": PATCH_SIZE * PATCH_SIZE,
            "burn_fraction": n_burn_pixels / (PATCH_SIZE * PATCH_SIZE),
        })
    return records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dataset(tmp_path):
    """Creates a synthetic patch dataset in a temp directory."""
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir()
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()

    n_patches = 20
    patch_ids = [f"fire_0001_{2015 + (i // 5)}-07-{(i % 28) + 1:02d}" for i in range(n_patches)]

    # Write .npz files
    for pid in patch_ids:
        make_synthetic_patch(pid, n_burn_pixels=400, out_dir=patches_dir)

    # Write manifests
    records = make_synthetic_manifest(patch_ids, n_burn_pixels=400, base_year=2015)
    for split in ["train", "val", "test"]:
        (splits_dir / f"{split}.json").write_text(json.dumps(records))

    # Write dummy stats
    stats = {ch: {"mean": 0.5, "std": 0.3} for ch in ["elevation", "slope_east", "slope_north",
        "wind_u", "wind_v", "temp_min", "temp_max", "humidity", "precip",
        "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "NDVI", "landcover",
        "PrevFireMask", "burnp3_burn_prob"]}
    (stats_dir / "channel_stats.json").write_text(json.dumps(stats))

    return {
        "patches_dir": patches_dir,
        "splits_dir": splits_dir,
        "stats_dir": stats_dir,
        "patch_ids": patch_ids,
        "records": records,
    }


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
        """Noise should NOT be applied to PrevFireMask (ch 17) or burnp3 (ch 18)."""
        from src.data.transforms import GaussianNoise
        X, y = self._make_tensors()
        X[17] = 0.5  # PrevFireMask
        X[18] = 0.3  # burnp3_burn_prob
        t = GaussianNoise(std=1.0)  # large std to make effect obvious
        X_out, _ = t(X, y)
        assert torch.allclose(X_out[17], X[17]), "PrevFireMask should not be noised"
        assert torch.allclose(X_out[18], X[18]), "burnp3_burn_prob should not be noised"

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
            "burnp3_burn_prob",
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


def test_synthetic_spread_follows_diagonal_wind():
    from src.data.assemble import generate_daily_masks
    final = np.ones((41, 41), dtype=np.float32)
    # Wind from the south-west pushes the fire north-east (up and right in the array)
    day1 = generate_daily_masks(final, 3, [40.0] * 3, [15.0] * 3, [225.0] * 3, [10.0] * 3)[1]
    assert day1[20 - 3, 20 + 3] and day1[20 + 3, 20 - 3]          # along the wind axis
    assert not day1[20 - 3, 20 - 3] and not day1[20 + 3, 20 + 3]  # across it
