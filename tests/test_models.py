"""
Smoke tests for the models, loss and training loop.

They catch shape and channel-count breakage before a multi-hour training run.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.utils.config import N_CHANNELS, PATCH_SIZE

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load_config(name: str):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name=name)


@pytest.mark.parametrize("name, x_shape", [
    ("unet", (2, N_CHANNELS, PATCH_SIZE, PATCH_SIZE)),
    ("convlstm", (2, 3, N_CHANNELS, PATCH_SIZE, PATCH_SIZE)),
])
def test_config_builds_a_model_that_trains_on_every_channel(name, x_shape):
    from src.models.losses import CombinedLoss
    from src.training.train import build_model

    model = build_model(load_config(name))
    logits = model(torch.randn(x_shape))
    assert logits.shape == (2, 1, PATCH_SIZE, PATCH_SIZE)

    loss = CombinedLoss()(logits, (torch.rand(2, PATCH_SIZE, PATCH_SIZE) > 0.9).float())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_one_epoch_trains_and_evaluates(synthetic_dataset):
    from src.data.dataset import WildfireDataset
    from src.models.losses import CombinedLoss
    from src.models.unet import build_unet
    from src.training.train import evaluate_epoch, train_one_epoch

    ds = synthetic_dataset
    dataset = WildfireDataset(
        manifest_path=ds["splits_dir"] / "train.json",
        patches_dir=ds["patches_dir"],
        stats_path=ds["stats_dir"] / "channel_stats.json",
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    model = build_unet(in_channels=N_CHANNELS, base_filters=8, depth=2)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    device = torch.device("cpu")

    assert np.isfinite(train_one_epoch(model, loader, optimizer, criterion, device))
    loss, metrics = evaluate_epoch(model, loader, criterion, device)
    assert np.isfinite(loss)
    assert 0.0 <= metrics["auc_pr"] <= 1.0
    assert "new_burn_auc_pr" in metrics


def test_new_burn_loss_ignores_already_burned_pixels():
    from src.models.losses import CombinedLoss

    criterion = CombinedLoss(pos_weight=5.0)
    logits = torch.randn(2, 1, 8, 8)
    targets = (torch.rand(2, 8, 8) > 0.7).float()
    mask = torch.ones(2, 8, 8)
    mask[:, :4] = 0  # top half already burned
    changed = logits.clone()
    changed[:, :, :4] = 10.0  # anything there must not matter
    assert torch.allclose(criterion(logits, targets, mask), criterion(changed, targets, mask))
    assert not torch.allclose(criterion(logits, targets), criterion(changed, targets))


@pytest.mark.parametrize("loss_on", ["all", "new_burn"])
def test_one_epoch_trains_with_either_loss(synthetic_dataset, loss_on):
    from src.data.dataset import WildfireDataset
    from src.models.losses import CombinedLoss
    from src.models.unet import build_unet
    from src.training.train import evaluate_epoch, train_one_epoch

    ds = synthetic_dataset
    dataset = WildfireDataset(
        ds["splits_dir"] / "train.json", ds["patches_dir"], ds["stats_dir"] / "channel_stats.json"
    )
    _, pos_weight = dataset.pixel_class_weights(new_burn_only=loss_on == "new_burn")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    model = build_unet(in_channels=N_CHANNELS, base_filters=8, depth=2)
    criterion = CombinedLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters())
    new_burn_only = loss_on == "new_burn"
    cpu = torch.device("cpu")
    assert np.isfinite(train_one_epoch(model, loader, optimizer, criterion, cpu, new_burn_only=new_burn_only))
    loss, metrics = evaluate_epoch(model, loader, criterion, cpu, new_burn_only)
    assert np.isfinite(loss) and 0.0 <= metrics["new_burn_auc_pr"] <= 1.0

