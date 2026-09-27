"""Environment configuration. Values are read per request so tests can patch env."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

BC_WEST = -139.06
BC_SOUTH = 48.30
BC_EAST = -114.03
BC_NORTH = 60.00

DEFAULT_SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
ALLOWED_SOURCES = frozenset(DEFAULT_SOURCES)

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov"
CWFIS_WFS_URL = "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows"

_MODEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _int_env(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


@dataclass(frozen=True)
class Settings:
    firms_map_key: str
    firms_sources: tuple[str, ...]
    firms_day_range: int
    hotspot_cache_seconds: int
    weather_cache_seconds: int
    request_timeout: float
    cors_origins: tuple[str, ...]
    xai_api_key: str
    grok_model: str
    firms_base_url: str
    cwfis_wfs_url: str


def get_settings() -> Settings:
    raw_sources = os.environ.get("FIRMS_SOURCES", "").strip()
    if raw_sources:
        requested = tuple(part.strip() for part in raw_sources.split(",") if part.strip())
        sources = tuple(source for source in requested if source in ALLOWED_SOURCES)
        if not sources:
            sources = DEFAULT_SOURCES
    else:
        sources = DEFAULT_SOURCES

    origins_raw = os.environ.get("CORS_ORIGINS", "").strip()
    if not origins_raw:
        origins = ("http://localhost:3000", "http://127.0.0.1:3000")
    else:
        origins = tuple(part.strip() for part in origins_raw.split(",") if part.strip()) or (
            "http://localhost:3000",
        )

    model = os.environ.get("GROK_MODEL", "grok-3").strip() or "grok-3"
    if not _MODEL_RE.match(model):
        model = "grok-3"

    return Settings(
        firms_map_key=os.environ.get("FIRMS_MAP_KEY", "").strip(),
        firms_sources=sources,
        firms_day_range=_int_env("FIRMS_DAY_RANGE", 2, 1, 5),
        hotspot_cache_seconds=_int_env("HOTSPOT_CACHE_SECONDS", 600, 0, 3600),
        weather_cache_seconds=_int_env("WEATHER_CACHE_SECONDS", 1800, 0, 7200),
        request_timeout=float(_int_env("HTTP_TIMEOUT_SECONDS", 30, 5, 120)),
        cors_origins=origins,
        xai_api_key=os.environ.get("XAI_API_KEY", "").strip(),
        grok_model=model,
        firms_base_url=os.environ.get("FIRMS_BASE_URL", FIRMS_BASE_URL).rstrip("/"),
        cwfis_wfs_url=os.environ.get("CWFIS_WFS_URL", CWFIS_WFS_URL).strip() or CWFIS_WFS_URL,
    )
