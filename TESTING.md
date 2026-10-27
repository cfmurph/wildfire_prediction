# Wildfire test plan

This file is the test plan for the map API and the Next.js map, plus the results of the suite added after PR #6 (`6006bee`).

## What actually ships

Two FastAPI apps live in this tree. They are not the same service.

| | Documented map API | Process that Docker and Railway start |
| --- | --- | --- |
| Module | `app.main:app` (`api/app/`) | `api.main:app` (`api/main.py`, `api/routers/`, `api/services/`) |
| How to run it locally | `PYTHONPATH=api:. uvicorn app.main:app` | `uvicorn api.main:app` |
| Routes | `GET /health`, `GET /hotspots`, `GET /weather`, `POST /report`, `GET /history`, `GET /risk`, `GET /predict` (501) | `GET /health`, `GET /`, `/api/v1/fires/*`, `/api/v1/weather/fwi`, `/api/v1/predict/spread`, `/api/v1/risk/map`, `POST /api/v1/situation` |
| Frontend that calls it | Leaflet components (`FireMap`, `Panel`, `ComingSoon`, `lib/api.ts`) are in the repo and are not mounted | `apps/web/app/page.tsx` mounts MapLibre (`WildfireMap`) and posts to `/api/v1/situation` |

`api/Dockerfile` and `railway.toml` both start `uvicorn api.main:app`. The image build context is the repository root. `COPY` must use `api/requirements.txt`; a root-level `requirements.txt` does not exist.

The older package is left in place. It is not dead in production. The `api/app` package is what the README and `api/pytest.ini` describe, and it is what a local `app.main:app` process serves. Tests cover both. Nothing from either package was deleted.

## Plan

Default runs mock NASA FIRMS, CWFIS, and xAI. `api/api_tests/conftest.py` clears hotspot and weather caches, removes `FIRMS_MAP_KEY`, `XAI_API_KEY`, `WILDFIRE_LIVE_SMOKE`, and `WILDFIRE_LIVE_GROK`, and replaces `app.firms.fetch_text`, `app.weather.fetch_json`, `httpx.AsyncClient`, and the OpenAI clients with blockers. Synchronous `httpx.Client` is not patched globally because Starlette's `TestClient` needs it. Live tests are a separate opt-in.

### Documented API (`api/app`)

- FIRMS CSV: empty body, missing header, malformed rows, missing columns, points outside the BC bbox, duplicate detections, confidence and day/night labels, acquisition clocks (`930` pads to `09:30`; hour 24 and bad dates are dropped).
- `FIRMS_MAP_KEY` absent or rejected by an upstream body that says the key is invalid (`503`). Other upstream 4xx/5xx and timeouts become `502`. The map key is never written to the response or the log message.
- Multiple VIIRS sources, `FIRMS_DAY_RANGE` bounds, and invalid day-range values.
- Hotspot cache: TTL hit, miss, expiry, and a separate weather cache. TTL `0` does not keep a failure.
- CWFIS: soft-fail on HTTP errors, non-JSON, HTML, arrays, and bad features. A failed first window does not invent a wider one.
- Clustering and geo helpers: 0.2° cells on negative longitudes, FRP ordering, haversine, compass labels (half-up, so 22.5° is NE).
- `POST /report`: missing or invalid cluster body, extra fields ignored, oversized or non-numeric confidence, missing `XAI_API_KEY`, Grok errors and timeouts. Spread `p25`/`p50`/`p75` stay `0`. The prompt in `prompts/live_situation_report.txt` forbids treating those zeros as a forecast.
- `GET /health` reports whether keys are configured and does not echo the key values.
- `GET /history`, `/risk`, and `/predict` return `501` with `coming_soon`. `POST` on those paths is `405`.
- CORS follows `CORS_ORIGINS` captured at import (default, an explicit origin, and `*`).
- Every variable in `api/.env.example` is parsed, including invalid model names, origins, and cache bounds.

### Shipped `/api/v1` API

- Deploy files parse: `railway.toml` is valid TOML, the health check is `/health`, and the Dockerfile copies a requirements file that exists.
- Hotspots: empty without a key, `days` outside 1–10 is `422`, bad rows and out-of-bbox points are dropped, a `500` or connect error returns an empty `FeatureCollection`, and the map key is not in the body or the log.
- Active fires and history keep valid features when one feature is bad. History years outside the supported range are `422`.
- Predict and risk responses match the heuristic functions that implement them (`climatology_v1`, percentiles present). They are not treated as a trained model.
- Situation reports JSON-encode fire names (so a name cannot break out of the payload) and do not return provider exception text.

### Frontend

Vitest and React Testing Library, jsdom, Leaflet mocked:

- `lib/format.ts`, `lib/utils.ts` (the two FWI scales are tested as they are written), `lib/api.ts`, `lib/views.ts`.
- `ViewSwitcher` (current, history year slider, predictions, risk).
- `ComingSoon` for the unmounted Leaflet views.
- `Panel`: the situation-report button is disabled when `health.grok_configured` is false. This is the Leaflet panel. It is not mounted by `app/page.tsx`.
- `FirePanel`: empty selection, a successful `POST /api/v1/situation`, and the button disabled only while the request is in flight. This mounted panel does not read a Grok key.
- `FireMap` with Leaflet mocked.

Playwright (Chromium) loads the production `next start` server, stubs Esri, OpenStreetMap, and FIRMS WMS tiles, and stubs `/api/v1/**` with an empty `FeatureCollection`. It checks the header, Current / History (year slider at 2023) / Predictions / Risk, and that the page calls `/api/v1/`.

### Research

`tests/test_metrics.py` stays in the default run (pytest, numpy, scikit-learn). `tests/test_dataset.py` imports PyTorch at module level. CI runs it only when `import torch` succeeds.

### CI

`.github/workflows/test.yml` runs on `push` and `pull_request` to `main`:

1. API tests with coverage XML uploaded as `api-coverage`.
2. Research metric tests, and dataset tests when torch imports.
3. Web lint, unit tests, `npm run build`, `tsc --noEmit`, Playwright Chromium.

## How to run

```bash
python -m pip install -r api/requirements-dev.txt
python -m pytest -c api/pytest.ini --rootdir api api/api_tests

python -m pip install pytest numpy scikit-learn
python -m pytest tests/test_metrics.py -q

cd apps/web
npm ci
npm test
npm run lint
npm run build
npm run typecheck
npx playwright install --with-deps chromium
npm run test:e2e
```

Opt-in live smoke (not part of CI): `WILDFIRE_LIVE_SMOKE=1` calls CWFIS. The same flag plus `FIRMS_MAP_KEY` calls FIRMS. There is no live Grok test.

## Counts

| Suite | Before | After |
| --- | --- | --- |
| API (`api/api_tests/`) | 17 passed (`test_api.py` only) | 101 collected: 96 passed, 2 skipped, 3 xfailed |
| Research metrics (`tests/test_metrics.py`) | 14 passed | 14 passed |
| Research dataset (`tests/test_dataset.py`) | 16 tests, not run (torch not installed) | 16 tests, still not run here; CI skips them unless torch imports |
| Web unit (Vitest) | 0 | 24 passed |
| Web e2e (Playwright) | 0 | 1 passed |

API files after this change: `test_api.py` 17, `test_cache_weather_config.py` 28, `test_deployed_legacy_api.py` 19, `test_firms_edges.py` 16, `test_live_smoke.py` 2, `test_report_contract.py` 19.

## Coverage

Measured by `pytest` with `--cov=app --cov=api.main --cov=api.routers --cov=api.services --cov=src.response.grok_client` (same sources as `api/pytest.ini`).

| Package | Statements | Missed | Coverage |
| --- | --- | --- | --- |
| `api/app` (documented map API) | 573 | 9 | 98% |
| Shipped `api.main`, `api.routers`, `api.services`, and `src.response.grok_client`, combined with `api/app` | 1001 | 32 | 97% |

Residual misses are defensive branches: a FIRMS 4xx that is not an invalid-key body inside `fetch_text`, non-finite optional floats, an empty CSV row, `nearest_station` when every station was already skipped, non-numeric weather coercion, a JSON value that starts with `{` but is not an object, and Grok helpers that the mocked client does not reach.

## Bugs

Fixed in this change:

| Severity | Bug | Fix |
| --- | --- | --- |
| High | `apps/web/tsconfig.json` contained a second JSON object after the first closing brace, so `tsc` and `next build` could not parse it. | Rewritten as one config. |
| High | `railway.toml` repeated `[build]` and `[deploy]`. Python `tomllib` rejects duplicate tables, so the file did not parse. | Collapsed to one `[build]`, one `[deploy]`, and the existing `[[services]]` entry. The start command is still `api.main:app`. |
| High | `api/Dockerfile` copied `requirements.txt` from the repo root. The file is `api/requirements.txt`, so the image build failed. | `COPY api/requirements.txt`. |
| High | `apps/web` still imports `leaflet` and `react-leaflet`, but those dependencies had been removed, so typecheck of the unmounted map failed. | Dependencies restored. `app/page.tsx` still mounts MapLibre. |
| Medium | Legacy FIRMS logging interpolated the exception, and httpx includes the request URL, which contains the map key. | Logs the exception type name only. |
| Medium | Legacy Grok built the prompt with an f-string (a fire name could inject text) and returned exception text to the client. | Payload is `json.dumps`. The client-facing error is a fixed sentence. |
| Medium | `src.response.grok_client.GrokClient` put exception text in logs and in `RuntimeError`. | Logs the type name. The raised error does not include the provider message. |
| Medium | One bad FIRMS row, or one bad active-fire feature, failed the whole collection. Legacy hotspots also kept non-finite and out-of-bbox coordinates. | Bad rows are skipped. Out-of-bbox and non-finite points are dropped. |
| Low | A weather failure was cached for 60 seconds even when `WEATHER_CACHE_SECONDS` was 0. | Failure TTL uses the configured value, and `0` does not cache. |
| Low | Acquisition timestamps accepted hour 24 and dates that are not `%Y-%m-%d`. | Those values are dropped. `"930"` still becomes `09:30`. |
| Low | Compass labels used banker's rounding, so 22.5° was N. | Half-up rounding. 315° stays NW. |
| Low | Pydantic validated `confidence` and `satellites` before the field validator, so a bad confidence type was `422` instead of coerced. | Validators use `mode="before"`. |
| Low | `WildfireMap.tsx` imported `useState` and `fwiColor` and did not use them. | Imports removed. |

Open, asserted with `pytest.mark.xfail(strict=True)` so a real fix becomes an XPASS and fails CI until the test is updated:

| Severity | Bug | Test |
| --- | --- | --- |
| High | The container and Railway start `api.main:app`. The README, pytest pythonpath, and `api/app` are `app.main:app`. The mounted page talks to `/api/v1`. The Leaflet page from the pre-merge tree (`FireMap`, `Panel`, `ComingSoon`, Soon badges) is not mounted. | `test_container_starts_the_documented_app` |
| Medium | `GET /api/v1/weather/fwi` returns a fabricated 10×10 grid (`is_synthetic: true`, station name `Synthetic`) when CWFIS cannot be reached. | `test_legacy_weather_does_not_invent_stations_when_upstream_fails` |
| Medium | `POST /api/v1/situation` without `XAI_API_KEY` still describes a median 24h spread forecast from the heuristic model. `POST /report` on the documented API keeps spread probabilities at zero. | `test_legacy_situation_without_key_does_not_state_a_spread_forecast` |

Documented behavior that was not changed and is not marked xfail:

- The mounted `FirePanel` does not disable "Generate Situation Report" when no XAI key is configured. It only disables the button while a request is in flight. The unmounted Leaflet `Panel` does disable the button when `grok_configured` is false, and a unit test locks that behavior.
- `lib/format.ts` and `lib/utils.ts` use different FWI band cutovers (5/10/20/30 versus 5/12/20/30). Each module is tested against its own scale.
- The documented hotspot cache key is the single string `hotspots`, not a key per map key or day range.
- Missing numeric weather fields on `POST /report` are coerced to 0.
- Legacy `/api/v1/predict/spread` and `/api/v1/risk/map` are heuristics. Tests check that contract. They do not claim a trained model.

## Not verified here

- Live FIRMS. No `FIRMS_MAP_KEY` in this environment. `test_live_firms_area_csv_smoke` is skipped unless `WILDFIRE_LIVE_SMOKE=1` and a key are set. It does not print the response body.
- Live Grok. No `XAI_API_KEY`. The suite never calls xAI. A live call would be billed, and there is no opt-in test for it.
- Live CWFIS. `test_live_cwfis_wfs_returns_geojson` is skipped unless `WILDFIRE_LIVE_SMOKE=1`. CWFIS needs no key. The default run uses a mock.
- `tests/test_dataset.py` (16 tests). They import torch, which is not installed. CI skips them for the same reason.
- Clicking a real fire polygon in the browser. The Playwright test uses an empty feature collection, so it checks view switching and that `/api/v1/` is called. It does not draw a hotspot or open a MapLibre popup.
