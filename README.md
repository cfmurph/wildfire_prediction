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

## Web app (Milestone 1)

The map has four views. Only **Current** is live in this release: near-real-time VIIRS hotspots in British Columbia, CWFIS fire-weather stations, and an on-demand Grok situation report. History (2012–2023, time slider), Long-term risk (burn likelihood), and Short-term (next-day spread) are tabs marked “Soon”. The API reserves `GET /history`, `GET /risk`, and `GET /predict` (HTTP 501) so those views can be added in their own routers. No trained spread model is required. Spread probabilities are left unset on purpose, and the live prompt tells Grok not to invent a forecast.

### Run locally

```bash
# API — Python 3.11+, lightweight deps only
python -m venv .venv && source .venv/bin/activate
pip install -r api/requirements.txt
cp api/.env.example api/.env   # fill in FIRMS_MAP_KEY and XAI_API_KEY
set -a && source api/.env && set +a
PYTHONPATH=api:. uvicorn app.main:app --reload --port 8000

# Web
cd apps/web
cp .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000. The browser calls `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

`GET /health` reports whether keys are set, without revealing them. `GET /hotspots` needs `FIRMS_MAP_KEY` (cached about 10 minutes). `GET /weather` reads CWFIS station FWI and degrades softly if that service is down. `POST /report` needs `XAI_API_KEY` and uses `src/response/grok_client.py` with `prompts/live_situation_report.txt`.

```bash
pip install -r api/requirements-dev.txt
pytest -c api/pytest.ini --rootdir api api/api_tests
```

Root `pytest` still runs only the research tests under `tests/`.

### Deploy

**API on Railway.** Connect the repo. `railway.toml` builds `api/Dockerfile` with the repository root as context so the image can copy `src/` and `prompts/`. Do not set the service root to `api/`. The container listens on `PORT`. Health check: `GET /health`.

Set `FIRMS_MAP_KEY`, `XAI_API_KEY`, and `CORS_ORIGINS` to the Vercel origin (comma-separated; local default is `http://localhost:3000` and `http://127.0.0.1:3000`). CORS is read when the process starts. Optional: `GROK_MODEL`, `FIRMS_DAY_RANGE` (1–5), `FIRMS_SOURCES`, `HOTSPOT_CACHE_SECONDS`, `WEATHER_CACHE_SECONDS`. `POST /report` calls a billed API — add your own auth or rate limit before a wide public launch. The in-memory cache is per process.

**Web on Vercel.** Root directory `apps/web`, framework Next.js. Set `NEXT_PUBLIC_API_URL` to the public Railway URL with no trailing slash.

`apps/web/vercel.json` sets `ignoreCommand` to `git diff HEAD^ HEAD --quiet -- .`. Vercel runs that from `apps/web`, so a commit that does not touch the web app (ML-only changes under `src/`, `configs/`, `scripts/`, `tests/`) skips the web deployment. Exit 0 skips the build; exit 1 builds.

### Environment variables

| Variable | Service | Required | Purpose |
|---|---|---|---|
| `FIRMS_MAP_KEY` | API | for `/hotspots` | NASA FIRMS map key |
| `XAI_API_KEY` | API | for `/report` | xAI Grok key read by `GrokClient` |
| `CORS_ORIGINS` | API | no | Browser origins allowed to call the API |
| `NEXT_PUBLIC_API_URL` | Web | in production | API base URL |

Copy `api/.env.example` and `apps/web/.env.example`. They contain placeholders only. Do not commit real keys. `.env` files are gitignored; the examples are not.

---

## Project structure

```
wildfire_prediction/
├── data/
│   ├── raw/            # Downloaded source files (gitignored)
│   ├── processed/      # Assembled .npz patches + split manifests (gitignored)
│   └── external/       # Alberta/Saskatchewan (Phase 2)
├── notebooks/          # EDA + experiment notebooks
├── api/                # FastAPI service (Railway) — hotspots, weather, reports
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
