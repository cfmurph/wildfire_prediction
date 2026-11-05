"""Spread predictions in Postgres: written by scripts/predict_spread.py, read by the API."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

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
        conn.execute("INSERT INTO spread_run (date, issued, failed_fires) VALUES (%s, %s, %s)",
                     (collection["date"], collection["issued"], collection.get("failed_fires", [])))
        conn.cursor().executemany(
            "INSERT INTO spread_area (date, fire_number, band, geom) "
            "VALUES (%s, %s, %s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))",
            [(collection["date"], f["properties"]["fire_number"], f["properties"]["band"], json.dumps(f["geometry"]))
             for f in collection["features"]],
        )


def save_archive(database_url: str, date_str: str, files: dict[str, Path]) -> None:
    """Record which input files a day has in the archive."""
    with psycopg.connect(database_url) as conn:
        conn.cursor().executemany(
            "INSERT INTO input_archive (date, source, path, bytes) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (date, source) DO UPDATE "
            "SET path = EXCLUDED.path, bytes = EXCLUDED.bytes, archived_at = now()",
            [(date_str, source, str(path), path.stat().st_size) for source, path in files.items()],
        )


def unscored_runs(database_url: str, since: str, before: str) -> list[str]:
    """The days predicted in a range that have no score yet, oldest first."""
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT date::text FROM spread_run WHERE date >= %s AND date < %s AND scored_at IS NULL ORDER BY date",
            (since, before),
        ).fetchall()
    return [day for day, in rows]


def run_features(database_url: str, date_str: str) -> list[dict]:
    """A day's flagged areas as GeoJSON features."""
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(
            "SELECT ST_AsGeoJSON(geom)::json, band, fire_number FROM spread_area WHERE date = %s", (date_str,)
        ).fetchall()
    return [{"type": "Feature", "geometry": geom, "properties": {"band": band, "fire_number": number}}
            for geom, band, number in rows]


def save_scores(database_url: str, date_str: str, scores: list[dict]) -> None:
    """Store a day's scores by fire and mark it scored, all or nothing."""
    with psycopg.connect(database_url) as conn:
        conn.execute("DELETE FROM spread_score WHERE date = %s", (date_str,))
        conn.cursor().executemany(
            "INSERT INTO spread_score (date, fire_number, unburned_km2, new_burn_km2, flagged_highest_km2, "
            "flagged_elevated_km2, hit_highest_km2, hit_elevated_km2) VALUES (%(date)s, %(fire_number)s, "
            "%(unburned_km2)s, %(new_burn_km2)s, %(flagged_highest_km2)s, %(flagged_elevated_km2)s, "
            "%(hit_highest_km2)s, %(hit_elevated_km2)s)",
            [score | {"date": date_str} for score in scores],
        )
        conn.execute("UPDATE spread_run SET scored_at = now() WHERE date = %s", (date_str,))


def latest_run(database_url: str) -> tuple[date, str] | None:
    """The newest day predicted and its GeoJSON text, or None before the first run."""
    with psycopg.connect(database_url) as conn:
        return conn.execute(LATEST_RUN).fetchone()
