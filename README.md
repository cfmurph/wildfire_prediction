# Canadian Wildfire Spread Prediction

> **Phase 1** — Benchmark dataset construction and ML baseline for next-day wildfire spread prediction in British Columbia.

There is no published ML benchmark for Canadian wildfire spread prediction. This project builds one from open data and establishes reproducible baselines using Random Forest, U-Net CNN, and ConvLSTM models — with xAI/Grok powering the decision-support layer.

---

## Research question

> Given everything known about a wildfire in British Columbia at time *T*, can we probabilistically forecast its spatial evolution over the next 24 hours?

---

## Dataset

Assembled from open Canadian government and scientific sources:

| Layer | Source | Resolution |
|---|---|---|
| Fire perimeters | BC Wildfire Service / Open Gov BC | Vector |
| Weather (ERA5) | Copernicus CDS | ~28 km |
| Fire Weather Index | CEMS fire danger reanalysis (Copernicus EWDS) | ~28 km |
| Terrain | SRTM | 90 m |
| Land cover | NRCan Canada LC 2020 | 30 m |

Patches: **64 × 64 pixels at 1 km/pixel** centered on active fire perimeters.
Coverage: **BC fire seasons 2012–2023**.

Daily weather uses the local (PDT) day: temperature max/min, minimum relative humidity, 16:00 wind and 24 h precipitation. Two of the 19 input channels are still zero-filled: NDVI (MODIS HDF needs `pyhdf`) and the BurnP3 burn probability.

**Targets are synthetic.** Perimeters give only a fire's final extent, so `generate_daily_masks` grows each fire from a seed toward its final perimeter, with daily spread scaled by FWI, ISI and wind. Scores therefore measure how well a model learns that generator, not real fire behaviour. Real VIIRS progression targets are the next step.

---

## Model progression

| Model | Description | Target |
|---|---|---|
| Random Forest | Pixel-level baseline (19 features) | Establishes AUC-PR anchor |
| U-Net CNN | Spatial encoder-decoder | ≥ 5pp AUC-PR over RF |
| ConvLSTM | Spatiotemporal (T=5 days history) | ≥ 5pp AUC-PR over U-Net |

Primary metric: **new-burn AUC-PR**, scored only on pixels not burned at day T. All-pixel AUC-PR rewards copying the day-T fire mask (persistence alone scores about 0.85), so every model is compared against persistence and a distance-to-fire baseline:

```bash
python scripts/evaluate_checkpoints.py --split test
```

Current U-Net result (test, seeds 42 / 1 / 2): **0.447 ± 0.064** new-burn AUC-PR against 0.177 for the distance baseline. On the September data, where most weather channels were empty, it scored 0.559 ± 0.012. The drop is under investigation: flip and rotation augmentation does not rotate the wind direction channel, which now drives the synthetic spread.

---

## Setup

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -e ".[dev]"

# 3. Configure Copernicus access (one-time)
# Register at https://cds.climate.copernicus.eu and place your key in ~/.cdsapirc.
# FWI comes from EWDS with the same key: accept the licence on the
# "cems-fire-historical-v1" dataset page at https://ewds.climate.copernicus.eu

# 4. Download Canadian data
python scripts/download_canada.py --province BC --years 2012 2023

# 5. Assemble patches
python scripts/assemble_patches.py --province BC

# 6. Run EDA
jupyter notebook notebooks/01_eda.ipynb

# 7. Train (MLflow 3 needs the flag for the local file store)
export MLFLOW_ALLOW_FILE_STORE=true
python src/training/train.py --config-name baseline_rf
python src/training/train.py --config-name unet training.seed=42
python src/training/train.py --config-name convlstm

# 8. Start MLflow UI
mlflow ui --backend-store-uri experiments/mlruns
```

The package build backend is `setuptools.build_meta`. `pip install -e .` installs the research code (including PyTorch). The web API does not use that install.

---

## Web app

A MapLibre map of British Columbia with three views:

- **Current**: active BC fires, NASA FIRMS VIIRS hotspots (NOAA-20/21) and CWFIS fire weather. Each fire's panel shows the nearest station's weather, a 24 h spread estimate from a wind- and FWI-driven heuristic (`api/services/predictor.py`; the trained ConvLSTM is not wired in yet), and an optional Grok situation report.
- **History**: BC fire perimeters by year, 2006 to the current season.
- **Risk**: where fires tend to start in a given month, from the ignition model (`experiments/checkpoints/ignition_rf.joblib`), split into lightning- and human-caused layers. Its probabilities are not calibrated, so the map shows them relative to the month's highest value.

### Run locally

```bash
# API: Python 3.11+, lightweight deps only (no PyTorch)
python -m venv .venv && source .venv/bin/activate
pip install -r api/requirements.txt
cp api/.env.example api/.env   # fill in FIRMS_MAP_KEY and XAI_API_KEY
set -a && source api/.env && set +a
uvicorn api.main:app --reload --port 8000

# Web
cd apps/web
cp .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. API docs are at http://localhost:8000/docs.

### Tests

```bash
pip install -r api/requirements-dev.txt
pytest -c api/pytest.ini --rootdir api api/api_tests   # API (network and Grok are mocked)
pytest tests/                                         # research
cd apps/web && npm test && npm run test:e2e           # web unit + Playwright
```

CI (`.github/workflows/test.yml`) runs the API and web suites on every PR to `main`. For research code it runs only `tests/test_metrics.py`; `tests/test_dataset.py` needs PyTorch, which CI does not install, so run it locally.

### Deploy

**API on Railway.** `railway.toml` builds `api/Dockerfile` with the repository root as context, so the image includes `src/`, the ignition model, and the climatology stats. Do not set the service root to `api/`. Health check: `GET /health`.

**Web on Vercel.** Root directory `apps/web`, framework Next.js. `apps/web/vercel.json` skips the build when a commit does not touch `apps/web`.

### Environment variables

| Variable | Service | Required | Purpose |
|---|---|---|---|
| `FIRMS_MAP_KEY` | API | for hotspots | NASA FIRMS map key |
| `XAI_API_KEY` | API | for situation reports | xAI Grok key |
| `CORS_ORIGINS` | API | in production | Comma-separated browser origins (the Vercel URL). Defaults to `http://localhost:3000` |
| `NEXT_PUBLIC_API_URL` | Web | in production | Railway API URL, no trailing slash |

`POST /api/v1/situation` calls the billed Grok API and is capped at 60 reports per hour per API process.

---

## Project structure

```
wildfire_prediction/
├── data/
│   ├── raw/            # Downloaded source files (gitignored)
│   ├── processed/      # Assembled .npz patches + split manifests (gitignored, except two stats files the API reads)
│   └── external/       # Alberta/Saskatchewan (Phase 2)
├── notebooks/          # EDA + experiment notebooks
├── api/                # FastAPI service (Railway): fires, weather, spread, risk, ignition, Grok reports
├── apps/web/           # Next.js map (Vercel)
├── src/
│   ├── data/           # Patch assembly + PyTorch Dataset + augmentation
│   ├── models/         # RF, U-Net, ConvLSTM
│   ├── training/       # Training loop + evaluation
│   ├── response/       # xAI Grok decision-support client
│   └── utils/          # Metrics, visualization, config
├── configs/            # Hydra YAML configs
├── prompts/            # Grok prompt templates
├── experiments/        # MLflow tracking root
├── scripts/            # Data download, patch assembly, checkpoint evaluation
└── tests/              # Research tests (API tests live in api/api_tests)
```

---

## Data sources

- [BC Wildfire Service open data](https://catalogue.data.gov.bc.ca/dataset/fire-perimeters-historical)
- [ERA5 / Copernicus CDS](https://cds.climate.copernicus.eu)
- [CEMS fire danger reanalysis / Copernicus EWDS](https://ewds.climate.copernicus.eu)
- [SRTM 90 m](https://srtm.csi.cgiar.org/)
- [Canada Land Cover 2020](https://open.canada.ca/data/en/dataset/4e615eae-b90c-420b-adee-2ca35896caf6)
- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) and [CWFIS Datamart](https://cwfis.cfs.nrcan.gc.ca/datamart) (web app: live hotspots and fire weather)
- [MODIS MOD13A2 NDVI](https://lpdaac.usgs.gov/products/mod13a2v061/) (not yet loaded)

---

## References

- Huot et al. (2022). *Next Day Wildfire Spread: A Machine Learning Data Set to Predict Wildfire Spreading from Remote-Sensing Data*. IEEE TGRS. [Google Research](https://research.google/pubs/next-day-wildfire-spread-a-machine-learning-dataset-to-predict-wildfire-spreading-from-remote-sensing-data/)
- Natural Resources Canada. [Canadian Wildland Fire Information System](https://cwfis.cfs.nrcan.gc.ca/)
- BC Wildfire Service. [Historical Fire Data](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/about-bcws/wildfire-history)

---

## License

MIT. Data is sourced from open government and scientific repositories under their respective licenses.
