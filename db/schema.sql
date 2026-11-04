-- The private store behind the daily jobs. Safe to re-run: psql wildfire -f db/schema.sql
CREATE EXTENSION IF NOT EXISTS postgis;

-- One row per day predicted, including days with nothing flagged, so "nothing burning" differs from "no run"
CREATE TABLE IF NOT EXISTS spread_run (
  date date PRIMARY KEY,
  issued timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS spread_area (
  date date NOT NULL REFERENCES spread_run ON DELETE CASCADE,
  fire_number text NOT NULL,
  band text NOT NULL CHECK (band IN ('highest', 'elevated')),
  geom geometry(Geometry, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS spread_area_date ON spread_area (date);
