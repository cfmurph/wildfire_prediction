"""
BC Wildfire Service active and historical fire data.

Active fires: BC ArcGIS public FeatureServer (no auth, live)
Historical:   BC Data Catalogue perimeters (local download or public WFS)
"""

from __future__ import annotations

import logging
from typing import Optional

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

        # Normalise property names for frontend
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            # Ensure a consistent set of fields
            feat["properties"] = {
                "fire_number":   props.get("FIRE_NUMBER") or props.get("fire_number", ""),
                "fire_name":     props.get("FIRE_NAME") or props.get("fire_name", ""),
                "fire_year":     props.get("FIRE_YEAR") or props.get("fire_year", ""),
                "size_ha":       float(props.get("CURRENT_SIZE") or props.get("size_ha") or 0),
                "fire_cause":    props.get("FIRE_CAUSE") or props.get("fire_cause", ""),
                "fire_status":   props.get("FIRE_STATUS") or props.get("fire_status", ""),
                "stage_of_control": props.get("STAGE_OF_CONTROL") or "",
                "ignition_date": props.get("IGNITION_DATE") or "",
                "lat":           feat["geometry"]["coordinates"][1] if feat.get("geometry") else 0,
                "lon":           feat["geometry"]["coordinates"][0] if feat.get("geometry") else 0,
            }

        return data

    except Exception as exc:
        log.error(f"BC active fires fetch failed: {exc}")
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

        for feat in data.get("features", []):
            props = feat.get("properties", {})
            feat["properties"] = {
                "fire_number": props.get("FIRE_NUMBER", ""),
                "fire_year":   int(props.get("FIRE_YEAR") or year),
                "size_ha":     float(props.get("FIRE_SIZE_HECTARES") or 0),
                "fire_cause":  props.get("FIRE_CAUSE", ""),
                "ignition_date": props.get("IGN_DATE", ""),
                "out_date":    props.get("OUT_DATE", ""),
            }

        return data

    except Exception as exc:
        log.error(f"BC historical perimeters fetch failed for {year}: {exc}")
        return {"type": "FeatureCollection", "features": []}
