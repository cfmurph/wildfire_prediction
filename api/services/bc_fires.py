"""
BC Wildfire Service active and historical fire data.

Active fires: BC ArcGIS public FeatureServer (no auth, live)
Historical:   BC Data Catalogue perimeters (local download or public WFS)
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

# BC Wildfire Service public ArcGIS FeatureServer — current active fires
BC_ACTIVE_FIRES_URL = (
    "https://services6.arcgis.com/ubm4tcTYICKBpist/arcgis/rest/services"
    "/BCWS_ActiveFires_PublicView/FeatureServer/0/query"
    "?where=1%3D1&outFields=*&outSR=4326&f=geojson"
)

# BC historical fire perimeters WFS (fixed URL)
BC_HISTORICAL_WFS = (
    "https://openmaps.gov.bc.ca/geo/pub/wfs?"
    "service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP"
    "&outputFormat=application%2Fjson&srsName=EPSG%3A4326"
    "&CQL_FILTER=FIRE_YEAR%3D{year}"
    "&count=2000"
)


async def get_active_fires() -> dict:
    """
    Fetch current active BC wildfires from the public ArcGIS FeatureServer.
    Returns GeoJSON FeatureCollection (always works, no auth required).
    """
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(BC_ACTIVE_FIRES_URL)
            resp.raise_for_status()
            data = resp.json()

        kept = []
        for feat in data.get("features") or []:
            if not isinstance(feat, dict):
                continue
            point = _point_lat_lon(feat)
            if point is None:
                continue
            lat, lon = point
            props = feat.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            feat["properties"] = {
                "fire_number": props.get("FIRE_NUMBER") or props.get("fire_number", ""),
                "fire_name": props.get("FIRE_NAME") or props.get("fire_name", ""),
                "fire_year": props.get("FIRE_YEAR") or props.get("fire_year", ""),
                "size_ha": _as_float(props.get("CURRENT_SIZE") or props.get("size_ha")),
                "fire_cause": props.get("FIRE_CAUSE") or props.get("fire_cause", ""),
                "fire_status": props.get("FIRE_STATUS") or props.get("fire_status", ""),
                "stage_of_control": props.get("STAGE_OF_CONTROL") or "",
                "ignition_date": props.get("IGNITION_DATE") or "",
                "lat": lat,
                "lon": lon,
            }
            kept.append(feat)
        data["features"] = kept
        return data

    except Exception as exc:
        log.error("BC active fires fetch failed (%s)", type(exc).__name__)
        return {"type": "FeatureCollection", "features": []}


async def get_historical_perimeters(year: int) -> dict:
    """
    Fetch BC historical fire perimeters for a given year.
    Uses the public WFS endpoint — returns up to 2000 features.
    """
    url = BC_HISTORICAL_WFS.format(year=year)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        kept = []
        for feat in data.get("features") or []:
            if not isinstance(feat, dict):
                continue
            props = feat.get("properties") or {}
            if not isinstance(props, dict):
                props = {}
            try:
                fire_year = int(props.get("FIRE_YEAR") or year)
            except (TypeError, ValueError):
                fire_year = year
            feat["properties"] = {
                "fire_number": props.get("FIRE_NUMBER", ""),
                "fire_year": fire_year,
                "size_ha": _as_float(props.get("FIRE_SIZE_HECTARES")),
                "fire_cause": props.get("FIRE_CAUSE", ""),
                "ignition_date": props.get("IGN_DATE", ""),
                "out_date": props.get("OUT_DATE", ""),
            }
            kept.append(feat)
        data["features"] = kept
        return data

    except Exception as exc:
        log.error("BC historical perimeters fetch failed for %s (%s)", year, type(exc).__name__)
        return {"type": "FeatureCollection", "features": []}


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _point_lat_lon(feat: dict) -> tuple[float, float] | None:
    geom = feat.get("geometry")
    if not isinstance(geom, dict) or geom.get("type") != "Point":
        return None
    coords = geom.get("coordinates")
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    lon = _as_float(coords[0], default=float("nan"))
    lat = _as_float(coords[1], default=float("nan"))
    if lat != lat or lon != lon:
        return None
    return lat, lon
