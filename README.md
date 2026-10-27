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
| Active hotspots | CWFIS / NASA FIRMS | 375 m |
| Weather (ERA5) | Copernicus CDS | ~28 km |
| Fire Weather Index | CWFIS | ~10 km |
| Terrain (CDEM) | NRCan | 20 m |
| Vegetation (NDVI) | MODIS MOD13A2 | 1 km |
| Land cover | NRCan Canada LC 2020 | 30 m |

Patches: **64 × 64 pixels at 1 km/pixel** centered on active fire perimeters.
Coverage: **BC fire seasons 2012–2023**.

---

## Model progression

| Model | Description | Target |
|---|---|---|
| Random Forest | Pixel-level baseline (18 features) | Establishes AUC-PR anchor |
| U-Net CNN | Spatial encoder-decoder | ≥ 5pp AUC-PR over RF |
| ConvLSTM | Spatiotemporal (T=5 days history) | ≥ 5pp AUC-PR over U-Net |

Primary metric: **AUC-PR** (appropriate for imbalanced burn/non-burn classes).

---

## Setup

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -e ".[dev]"

# 3. Configure ERA5 access (one-time)
# Register at https://cds.climate.copernicus.eu and place your key in ~/.cdsapirc

# 4. Download Canadian data
python scripts/download_canada.py --province BC --years 2012 2023

# 5. Assemble patches
python scripts/assemble_patches.py --province BC

# 6. Run EDA
jupyter notebook notebooks/01_eda.ipynb

# 7. Train baseline
python src/training/train.py --config-name baseline_rf

# 8. Start MLflow UI
mlflow ui --backend-store-uri experiments/mlruns
```

The package build backend is `setuptools.build_meta`. `pip install -e .` installs the research code (including PyTorch). The web API does not use that install.

---

## Web app

A MapLibre map of British Columbia with four views:

- **Current**: active BC fires, NASA FIRMS VIIRS hotspots (NOAA-20/21), CWFIS fire weather, and an on-demand Grok situation report per fire.
- **History**: BC fire perimeters by year (2012–2023).
- **Predictions**: next-day spread ellipses from a wind- and FWI-driven heuristic (`api/services/predictor.py`). The trained ConvLSTM is not wired in yet.
- **Risk**: monthly FWI climatology plus the ignition model (`experiments/checkpoints/ignition_rf.joblib`), split into lightning- and human-caused layers.

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

CI (`.github/workflows/tests.yml`) runs all three on every PR to `main`.

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
│   ├── processed/      # Assembled .npz patches + split manifests (gitignored)
│   └── external/       # Alberta/Saskatchewan (Phase 2)
├── notebooks/          # EDA + experiment notebooks
├── api/                # FastAPI service (Railway): fires, weather, spread, risk, ignition, Grok reports
├── apps/web/           # Next.js map (Vercel)
├── src/
│   ├── data/           # Download + assembly + PyTorch Dataset
│   ├── models/         # RF, U-Net, ConvLSTM
│   ├── training/       # Training loop + evaluation
│   ├── response/       # xAI Grok decision-support client
│   └── utils/          # Metrics, visualization, config
├── configs/            # Hydra YAML configs
├── prompts/            # Grok prompt templates
├── experiments/        # MLflow tracking root
├── scripts/            # Data download + experiment runner
└── tests/              # Research tests (API tests live in api/api_tests)
```

---

## Data sources

- [BC Wildfire Service open data](https://catalogue.data.gov.bc.ca/dataset/fire-perimeters-historical)
- [CWFIS Datamart](https://cwfis.cfs.nrcan.gc.ca/datamart)
- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)
- [ERA5 / Copernicus CDS](https://cds.climate.copernicus.eu)
- [Canadian Digital Elevation Model](https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333)
- [Canada Land Cover 2020](https://open.canada.ca/data/en/dataset/4e615eae-b90c-420b-adee-2ca35896caf6)
- [MODIS MOD13A2 NDVI](https://lpdaac.usgs.gov/products/mod13a2v061/)

---

## References

- Huot et al. (2022). *Next Day Wildfire Spread: A Machine Learning Data Set to Predict Wildfire Spreading from Remote-Sensing Data*. IEEE TGRS. [Google Research](https://research.google/pubs/next-day-wildfire-spread-a-machine-learning-dataset-to-predict-wildfire-spreading-from-remote-sensing-data/)
- Natural Resources Canada. [Canadian Wildland Fire Information System](https://cwfis.cfs.nrcan.gc.ca/)
- BC Wildfire Service. [Historical Fire Data](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/about-bcws/wildfire-history)

---

## License

MIT. Data is sourced from open government and scientific repositories under their respective licenses.
