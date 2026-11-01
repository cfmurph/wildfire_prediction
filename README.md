# Canadian Wildfire Spread Prediction

> **Phase 1** — Benchmark dataset construction and ML baseline for next-day wildfire spread prediction in British Columbia.

There is no published ML benchmark for Canadian wildfire spread prediction. This project builds one from open data and establishes reproducible baselines using Random Forest, U-Net CNN, and ConvLSTM models.

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

Daily weather uses the local (PDT) day: temperature max/min, minimum relative humidity, 16:00 wind and 24 h precipitation. NDVI comes from MODIS 16-day composites: run `scripts/prepare_ndvi.py` after downloading (it needs `pip install -e ".[data]"` for HDF4). Each patch uses the latest composite that ended by its date, so the input cannot show the next day's burn. `years_since_fire` gives the years since each pixel last burned, from BC fire perimeters back to 1972 (earlier seasons only, capped at 40).

**Targets come from satellite detections.** Perimeters give only a fire's final extent, so `firms_daily_masks` marks a pixel burned from the first day a NASA FIRMS VIIRS detection (Suomi NPP and NOAA-20, 375 m) falls in it. Only detections inside the fire's own final perimeter count, which drops other fires and non-wildfire burning such as slash piles. Cloud and smoke can hide burning, so some burned pixels are never detected.

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

Current U-Net result (test, seeds 42 / 1 / 2): **0.523 ± 0.016** new-burn AUC-PR (0.521 / 0.508 / 0.540) against 0.183 for the distance baseline. That is up from 0.447 ± 0.064 before terrain and wind were encoded as rotation-aware vectors and border pixels were filled; the September data, where most weather channels were empty, scored 0.559 ± 0.012. Two channels are still empty in the training split (NDVI and the Burn-P3 burn probability), so the model effectively sees 17 of 19.

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
# The FIRMS fire-detection archive (data/raw/firms) needs no key; MODIS goes back to 2000
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

**BC Wildfire Map**: a MapLibre map of British Columbia with three views:

- **Current**: active BC fires, NASA FIRMS VIIRS hotspots (NOAA-20/21) and CWFIS fire weather. Each fire's panel shows the nearest station's weather, a 24 h spread estimate from a wind- and FWI-driven heuristic (`api/services/predictor.py`; the trained ConvLSTM is not wired in yet).
- **History**: BC fire perimeters by year, 2006 to the current season.
- **Risk**: where fires typically start in a given month, split into lightning- and human-caused layers: recorded BC fire starts from the National Fire Database, 1996–2025, counted on a 5 km grid, smoothed over about 10 km and averaged per year. `scripts/compute_fire_starts.py` builds `data/processed/stats/fire_starts.json` from the points that `download_canada.py` fetches.

### Run locally

```bash
# API: Python 3.11+, lightweight deps only (no PyTorch)
python -m venv .venv && source .venv/bin/activate
pip install -r api/requirements.txt
cp api/.env.example api/.env   # fill in FIRMS_MAP_KEY
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
pytest -c api/pytest.ini --rootdir api api/api_tests   # API (network is mocked)
pytest tests/                                         # research
cd apps/web && npm test && npm run test:e2e           # web unit + Playwright
```

CI (`.github/workflows/test.yml`) runs the API, research (CPU PyTorch) and web suites on every PR to `main`.

### Deploy

One Railway service runs everything. `railway.toml` builds `api/Dockerfile` with the repository root as context: a Node stage builds the web app as static files (`output: "export"`), and the Python stage serves them from `/` next to the API under `/api/v1`, so there is one origin and no CORS. Do not set the service root to `api/`. Health check: `GET /health`.

Upstream feeds (BC Wildfire Service, FIRMS, CWFIS) are cached for 5 minutes per process, past seasons for a day, so visitor traffic does not reach them. Responses are gzip-compressed.

To check the image locally:

```bash
docker build -f api/Dockerfile -t bc-wildfire-map .
docker run --rm -p 8000:8000 -e FIRMS_MAP_KEY=... bc-wildfire-map   # http://localhost:8000
```

### Environment variables

| Variable | Service | Required | Purpose |
|---|---|---|---|
| `FIRMS_MAP_KEY` | API | for hotspots | NASA FIRMS map key; without it the map shows no hotspots |
| `CORS_ORIGINS` | API | local dev only | Browser origins allowed to call the API from `next dev`. Defaults to `http://localhost:3000` |
| `NEXT_PUBLIC_API_URL` | Web | local dev only | API for `next dev`; defaults to `http://localhost:8000`. Production builds call their own origin |

---

## Project structure

```
wildfire_prediction/
├── data/
│   ├── raw/            # Downloaded source files (gitignored)
│   ├── processed/      # Assembled .npz patches + split manifests (gitignored, except stats/fire_starts.json, which the API serves)
│   └── external/       # Alberta/Saskatchewan (Phase 2)
├── notebooks/          # EDA + experiment notebooks
├── api/                # FastAPI service (Railway): fires, weather, spread, fire starts; serves the web app
├── apps/web/           # Next.js map, exported as static files
├── src/
│   ├── data/           # Patch assembly + PyTorch Dataset + augmentation
│   ├── models/         # RF, U-Net, ConvLSTM
│   ├── training/       # Training loop + evaluation
│   └── utils/          # Metrics, visualization, config
├── configs/            # Hydra YAML configs
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
