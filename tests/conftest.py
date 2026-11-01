"""Shared fixtures: synthetic .npz patches and manifests, so no real data is required."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.config import CHANNELS, N_CHANNELS, PATCH_SIZE

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
    stats = {ch: {"mean": 0.5, "std": 0.3} for ch in CHANNELS}
    (stats_dir / "channel_stats.json").write_text(json.dumps(stats))

    return {
        "patches_dir": patches_dir,
        "splits_dir": splits_dir,
        "stats_dir": stats_dir,
        "patch_ids": patch_ids,
        "records": records,
    }
