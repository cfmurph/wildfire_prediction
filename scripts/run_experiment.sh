#!/usr/bin/env bash
# Run the full Phase 1 experiment sequence.
# Assumes: data downloaded and patches assembled.
# Usage: bash scripts/run_experiment.sh

set -euo pipefail

echo "=== Canadian Wildfire Spread Prediction — Phase 1 Experiments ==="
echo ""

# ---- 1. RF Baseline ----
echo "[1/4] Training Random Forest baseline…"
python -c "
from src.models.baseline_rf import train_and_evaluate
from pathlib import Path
train_and_evaluate(
    patches_dir=Path('data/processed/patches'),
    splits_dir=Path('data/processed/splits'),
    output_dir=Path('experiments/checkpoints'),
    mlflow_tracking_uri='experiments/mlruns',
)
"

# ---- 2. U-Net CNN ----
echo "[2/4] Training U-Net CNN…"
python src/training/train.py --config-name unet

# ---- 3. ConvLSTM T=1 (ablation) ----
echo "[3/4] Training ConvLSTM (T=1, ablation)…"
python src/training/train.py --config-name convlstm "data.seq_len=1"

# ---- 4. ConvLSTM T=5 ----
echo "[4/4] Training ConvLSTM (T=5)…"
python src/training/train.py --config-name convlstm "data.seq_len=5"

echo ""
echo "=== All experiments complete ==="
echo "View results: mlflow ui --backend-store-uri experiments/mlruns"
