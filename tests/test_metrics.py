"""
Tests for src/utils/metrics.py

Verifies metric implementations against known values and edge cases.
"""

import numpy as np
import pytest
from src.utils.metrics import compute_metrics, compare_models, calibration_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_predictions():
    """Perfect binary predictions — all metrics should be 1.0 except Brier=0."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=(64, 64)).astype(np.float32)
    y_prob = y_true.copy()  # perfect probabilities
    return y_true, y_prob


@pytest.fixture
def random_predictions():
    """Random predictions — AUC-PR should be near burn_fraction, AUC-ROC near 0.5."""
    rng = np.random.default_rng(0)
    y_true = (rng.random((64, 64)) < 0.15).astype(np.float32)  # ~15% burn
    y_prob = rng.random((64, 64)).astype(np.float32)
    return y_true, y_prob


@pytest.fixture
def imbalanced_predictions():
    """Typical wildfire scenario: ~15% burning pixels."""
    rng = np.random.default_rng(7)
    y_true = (rng.random((64, 64)) < 0.15).astype(np.float32)
    # Reasonably good model: high prob where burning, low elsewhere
    y_prob = y_true * 0.8 + rng.random((64, 64)) * 0.2
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_all_expected_keys(self, imbalanced_predictions):
        y_true, y_prob = imbalanced_predictions
        m = compute_metrics(y_true, y_prob)
        expected_keys = {"auc_pr", "auc_roc", "f1", "dice", "iou", "brier", "burn_fraction"}
        assert expected_keys.issubset(m.keys()), f"Missing keys: {expected_keys - m.keys()}"

    def test_all_values_are_floats(self, imbalanced_predictions):
        y_true, y_prob = imbalanced_predictions
        m = compute_metrics(y_true, y_prob)
        for k, v in m.items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"

    def test_perfect_predictions(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        m = compute_metrics(y_true, y_prob)
        assert m["auc_pr"] == pytest.approx(1.0, abs=1e-6)
        assert m["auc_roc"] == pytest.approx(1.0, abs=1e-6)
        assert m["f1"] == pytest.approx(1.0, abs=1e-6)
        assert m["dice"] == pytest.approx(1.0, abs=1e-6)
        assert m["iou"] == pytest.approx(1.0, abs=1e-6)
        assert m["brier"] == pytest.approx(0.0, abs=1e-6)

    def test_burn_fraction_correct(self, imbalanced_predictions):
        y_true, y_prob = imbalanced_predictions
        m = compute_metrics(y_true, y_prob)
        expected_fraction = float(y_true.mean())
        assert m["burn_fraction"] == pytest.approx(expected_fraction, abs=1e-4)

    def test_metrics_in_valid_range(self, random_predictions):
        y_true, y_prob = random_predictions
        m = compute_metrics(y_true, y_prob)
        for key in ["auc_pr", "auc_roc", "f1", "dice", "iou", "burn_fraction"]:
            assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0,1]"
        assert m["brier"] >= 0.0

    def test_accepts_arbitrary_shapes(self):
        """Metrics should work on 1D, 2D, and 3D arrays."""
        rng = np.random.default_rng(1)
        for shape in [(1000,), (32, 32), (4, 64, 64)]:
            y_true = (rng.random(shape) < 0.2).astype(np.float32)
            y_prob = rng.random(shape).astype(np.float32)
            m = compute_metrics(y_true, y_prob)
            assert "auc_pr" in m

    def test_all_zeros_target(self):
        """Degenerate case: no burning pixels → AUC metrics are NaN."""
        y_true = np.zeros((64, 64), dtype=np.float32)
        y_prob = np.random.rand(64, 64).astype(np.float32)
        m = compute_metrics(y_true, y_prob)
        assert np.isnan(m["auc_pr"])
        assert np.isnan(m["auc_roc"])

    def test_all_ones_target(self):
        """Degenerate case: all burning pixels."""
        y_true = np.ones((64, 64), dtype=np.float32)
        y_prob = np.random.rand(64, 64).astype(np.float32)
        m = compute_metrics(y_true, y_prob)
        assert np.isnan(m["auc_pr"])

    def test_threshold_parameter(self, imbalanced_predictions):
        """Changing threshold affects hard metrics (F1, Dice, IoU) but not AUC."""
        y_true, y_prob = imbalanced_predictions
        m_05 = compute_metrics(y_true, y_prob, threshold=0.5)
        m_09 = compute_metrics(y_true, y_prob, threshold=0.9)
        # AUC metrics are threshold-independent
        assert m_05["auc_pr"] == pytest.approx(m_09["auc_pr"], abs=1e-6)
        # Hard metrics change with threshold
        # (F1 at 0.9 threshold should generally be lower due to fewer predictions)
        assert m_05["f1"] != m_09["f1"] or m_05["f1"] == 0.0


# ---------------------------------------------------------------------------
# Dice and IoU relationship
# ---------------------------------------------------------------------------

class TestDiceIoURelationship:
    def test_dice_geq_iou(self, imbalanced_predictions):
        """Dice is always >= IoU for binary predictions: Dice = 2*IoU/(1+IoU)."""
        y_true, y_prob = imbalanced_predictions
        m = compute_metrics(y_true, y_prob)
        if m["iou"] > 0:
            expected_dice = 2 * m["iou"] / (1 + m["iou"])
            assert m["dice"] == pytest.approx(expected_dice, abs=0.01)


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

class TestCalibrationData:
    def test_output_shape(self, imbalanced_predictions):
        y_true, y_prob = imbalanced_predictions
        mean_pred, frac_pos = calibration_data(y_true, y_prob, n_bins=10)
        assert mean_pred.shape == (10,)
        assert frac_pos.shape == (10,)

    def test_perfect_calibration(self):
        """A perfectly calibrated model has frac_pos ≈ mean_pred in each bin."""
        rng = np.random.default_rng(42)
        # Generate calibrated probabilities
        y_prob = rng.random(10000).astype(np.float32)
        y_true = rng.random(10000).astype(np.float32) < y_prob
        y_true = y_true.astype(np.float32)
        mean_pred, frac_pos = calibration_data(y_true, y_prob, n_bins=10)
        valid = ~np.isnan(frac_pos)
        if valid.sum() > 0:
            # Should be roughly calibrated (within 0.1 per bin)
            np.testing.assert_allclose(
                frac_pos[valid], mean_pred[valid], atol=0.15
            )


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_string(self, imbalanced_predictions):
        y_true, y_prob = imbalanced_predictions
        m = compute_metrics(y_true, y_prob)
        table = compare_models({"model_a": m, "model_b": m})
        assert isinstance(table, str)
        assert "model_a" in table
        assert "auc_pr" in table

    def test_handles_empty(self):
        table = compare_models({})
        assert isinstance(table, str)
