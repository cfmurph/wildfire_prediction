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

Patches: **64 × 64 pixels at 1 km/pixel**, one per fire-day with new burning. Fires wider than 64 km are covered by a grid of adjacent patches.
Coverage: **BC fire seasons 2012–2024**: train 2012–2020, validation 2021–2022, test 2023 and 2024.

Daily weather uses the local (PDT) day: temperature max/min, minimum relative humidity, 16:00 wind and 24 h precipitation. NDVI comes from MODIS 16-day composites: run `scripts/prepare_ndvi.py` after downloading (it needs `pip install -e ".[data]"` for HDF4). Each patch uses the latest composite that ended by its date, so the input cannot show the next day's burn. `years_since_fire` gives the years since each pixel last burned, from BC fire perimeters back to 1972 (earlier seasons only, capped at 40).

**Targets come from satellite detections.** Perimeters give only a fire's final extent, so `firms_daily_masks` marks a pixel burned from the first day a NASA FIRMS VIIRS detection (Suomi NPP and NOAA-20, 375 m) falls in it. Only connected clusters of detections that reach into one of the season's mapped perimeters count: that keeps growth past a perimeter mapped before the fire stopped, and neighbouring fires in the same patch, while dropping isolated burning such as slash piles. Cloud and smoke can hide burning, so some burned pixels are never detected.

---

## Model progression

| Model | Description | Target |
|---|---|---|
| Random Forest | Pixel-level baseline (19 features) | Establishes AUC-PR anchor |
| U-Net CNN | Spatial encoder-decoder | ≥ 5pp AUC-PR over RF |
| ConvLSTM | Spatiotemporal (T=5 days history) | ≥ 5pp AUC-PR over U-Net |

Primary metric: **new-burn AUC-PR**, scored only on pixels not burned at day T. All-pixel AUC-PR rewards copying the day-T fire mask (persistence alone scores about 0.96), so every model is compared against persistence and a distance-to-fire baseline:

```bash
python scripts/evaluate_checkpoints.py --split test --unet seed42.pt seed1.pt seed2.pt   # one row per season
python scripts/channel_importance.py --checkpoint seed42.pt   # which inputs the model relies on
```

Current U-Net result on real (VIIRS) targets, seeds 42 / 1 / 2, new-burn AUC-PR: **0.087 / 0.088 / 0.090** in 2023 against 0.041 for the distance baseline, and **0.099 / 0.106 / 0.103** in 2024 against 0.038. Random guessing scores about 0.003. Validation (2021–22): 0.121 against 0.049. Every run is over twice the distance baseline; resampling the test fires puts the low end at about 1.9× in 2023 and 2.3× in 2024. Data: 18,097 patches from 1,170 fires; a typical patch gains 4 burned km² out of 4,096.

Earlier results (0.45–0.56) were on synthetic targets grown from final perimeters by an FWI- and wind-driven rule. They measured how well a model learns that rule and are not comparable.

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
python scripts/download_canada.py --province BC --years 2012 2024

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

- **Current**: active BC fires, NASA FIRMS VIIRS hotspots (NOAA-20/21) and CWFIS fire weather. Each fire's panel shows the nearest station's weather.
  During the fire season it also shades where each actively burning fire is most likely to grow next, from the U-Net (see *Live spread prediction*).
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

### Live spread prediction

`scripts/predict_spread.py` runs the trained U-Net on every BC fire that is burning, once a day, and writes it to Postgres and `data/live/spread.geojson`. The API serves it at `/api/v1/predict/spread-map` and the Current view draws as two bands (highest / elevated chance). It builds inputs like training does, from sources available on the day: FIRMS VIIRS near-real-time detections, ECMWF open-data weather and CWFIS fire-weather grids (`src/data/live.py`). It needs the research environment, `FIRMS_MAP_KEY` and the static raw data from `scripts/download_canada.py`:

```bash
python -m scripts.predict_spread   # day T = yesterday
```

The models are read from `experiments/checkpoints/spread/`. The `.pt` files are not in git, so back them up.

Run it each morning, after the night satellite pass. To do that on a Mac at 08:00 (or on waking, if it was asleep), logging to `data/live/spread.log`:

```bash
sed "s|REPO|$PWD|g" scripts/com.wildfire.spread.plist > ~/Library/LaunchAgents/com.wildfire.spread.plist
launchctl load ~/Library/LaunchAgents/com.wildfire.spread.plist
```

The predictions are kept in a private Postgres database with PostGIS, one row per day, so there is a history to score. To set it up on a Mac:

```bash
brew install postgresql@17 postgis
brew services start postgresql@17
createdb wildfire
psql wildfire -f db/schema.sql
echo 'DATABASE_URL=postgresql://localhost/wildfire' >> api/.env
```

Each run also keeps the day's inputs in `data/live/archive/<date>/`, listed in the `input_archive` table: the ECMWF forecast cropped to BC and out to the day being predicted, the CWFIS grids, FIRMS rows and the BCWS fire list. ECMWF deletes its forecasts after about 4 days, so a day not archived cannot be backtested later. After pulling a change to `db/schema.sql`, run `psql wildfire -f db/schema.sql` again.

A failed run, or one that could not predict a fire or archive a source, sends a macOS notification; a fire that failed is listed in `spread_run.failed_fires`. If no burning tile can be predicted the run stops and leaves the last result in place. A job that never starts, because the Mac was off, sends nothing.

Each morning also scores the previous prediction against that day's new detections, by fire in `spread_score` and by day in the `spread_score_daily` view. It needs the day's prediction and archived fire list, and catches up on days up to a week back. Detections are a lower bound on what burned, because cloud and smoke hide fire, and a day's score is noisy: judge the model over many days.

Browse it with pgAdmin: host `localhost`, port `5432`, database `wildfire`, your Mac username.

With `DATABASE_URL` set the API reads the database and stops serving a run three days after its date, so a stopped job clears the map layer. Without it the API serves the file, up to two days old: when running the Docker image, mount the output with `-v "$PWD/data/live:/app/data/live"`.

### Tests

```bash
pip install -r api/requirements-dev.txt
createdb wildfire_test && psql wildfire_test -f db/schema.sql    # once, for the database tests
TEST_DATABASE_URL=postgresql://localhost/wildfire_test pytest -c api/pytest.ini --rootdir api api/api_tests   # API (network is mocked)
pytest tests/                                         # research
cd apps/web && npm test && npm run test:e2e           # web unit + Playwright
```

CI (`.github/workflows/test.yml`) runs the API, research (CPU PyTorch) and web suites on every PR to `main`.

### Deploy

The app runs locally from one Docker image. `api/Dockerfile` builds with the repository root as context: a Node stage builds the web app as static files (`output: "export"`), and the Python stage serves them from `/` next to the API under `/api/v1`, so there is one origin and no CORS. Health check: `GET /health`.

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
| `DATABASE_URL` | API, daily job | for history | Postgres holding the spread predictions; without it the API serves the file |
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
├── api/                # FastAPI service: fires, weather, spread, fire starts; serves the web app
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
