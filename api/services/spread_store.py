"""Spread predictions in Postgres: written by scripts/predict_spread.py, read by the API."""

from __future__ import annotations

import json
from datetime import date

import psycopg

LATEST_RUN = """
SELECT r.date, json_build_object(
  'type', 'FeatureCollection',
  'date', r.date,
  'issued', r.issued,
  'features', coalesce((
    SELECT json_agg(json_build_object(
      'type', 'Feature',
      'geometry', ST_AsGeoJSON(a.geom)::json,
      'properties', json_build_object('band', a.band, 'fire_number', a.fire_number)
    )) FROM spread_area a WHERE a.date = r.date
  ), '[]'::json)
)::text
FROM spread_run r ORDER BY r.date DESC LIMIT 1
"""


def save_run(database_url: str, collection: dict) -> None:
    """Store a day's prediction, replacing any earlier run for that day, all or nothing."""
    with psycopg.connect(database_url) as conn:
        conn.execute("DELETE FROM spread_run WHERE date = %s", (collection["date"],))
        conn.execute("INSERT INTO spread_run VALUES (%s, %s)", (collection["date"], collection["issued"]))
        conn.cursor().executemany(
            "INSERT INTO spread_area (date, fire_number, band, geom) "
            "VALUES (%s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
            [(collection["date"], f["properties"]["fire_number"], f["properties"]["band"], json.dumps(f["geometry"]))
             for f in collection["features"]],
        )


def latest_run(database_url: str) -> tuple[date, str] | None:
    """The newest day predicted and its GeoJSON text, or None before the first run."""
    with psycopg.connect(database_url) as conn:
        return conn.execute(LATEST_RUN).fetchone()
