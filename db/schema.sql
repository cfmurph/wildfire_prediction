-- The private store behind the daily jobs. Safe to re-run: psql wildfire -f db/schema.sql
CREATE EXTENSION IF NOT EXISTS postgis;

-- One row per day predicted, including days with nothing flagged, so "nothing burning" differs from "no run"
CREATE TABLE IF NOT EXISTS spread_run (
  date date PRIMARY KEY,
  issued timestamptz NOT NULL
);

-- Fires with a tile that could not be predicted: no areas for them does not mean nothing flagged
ALTER TABLE spread_run ADD COLUMN IF NOT EXISTS failed_fires text[] NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS spread_area (
  date date NOT NULL REFERENCES spread_run ON DELETE CASCADE,
  fire_number text NOT NULL,
  band text NOT NULL CHECK (band IN ('highest', 'elevated')),
  geom geometry(Geometry, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS spread_area_date ON spread_area (date);

-- What the daily run kept of its inputs; the files themselves are in data/live/archive
CREATE TABLE IF NOT EXISTS input_archive (
  date date NOT NULL,
  source text NOT NULL,
  path text NOT NULL,
  bytes bigint NOT NULL,
  archived_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (date, source)
);

-- Each prediction against the next day's detections, by fire. Areas are 1 km pixels; a band's area excludes the other's
ALTER TABLE spread_run ADD COLUMN IF NOT EXISTS scored_at timestamptz;

CREATE TABLE IF NOT EXISTS spread_score (
  date date NOT NULL REFERENCES spread_run ON DELETE CASCADE,
  fire_number text NOT NULL,
  unburned_km2 integer NOT NULL,
  new_burn_km2 integer NOT NULL,
  flagged_highest_km2 integer NOT NULL,
  flagged_elevated_km2 integer NOT NULL,
  hit_highest_km2 integer NOT NULL,
  hit_elevated_km2 integer NOT NULL,
  PRIMARY KEY (date, fire_number)
);

-- In testing the highest band covered about 1% of unburned land and caught 40% of new burn; with elevated, 7% and 82%
CREATE OR REPLACE VIEW spread_score_daily AS
SELECT date,
  count(*) AS fires,
  sum(new_burn_km2) AS new_burn_km2,
  sum(flagged_highest_km2) AS flagged_highest_km2,
  sum(flagged_elevated_km2) AS flagged_elevated_km2,
  round(100.0 * sum(hit_highest_km2) / nullif(sum(new_burn_km2), 0), 1) AS caught_highest_pct,
  round(100.0 * sum(hit_highest_km2 + hit_elevated_km2) / nullif(sum(new_burn_km2), 0), 1) AS caught_both_pct,
  round(100.0 * sum(flagged_highest_km2) / nullif(sum(unburned_km2), 0), 1) AS land_highest_pct,
  round(100.0 * sum(flagged_highest_km2 + flagged_elevated_km2) / nullif(sum(unburned_km2), 0), 1) AS land_both_pct
FROM spread_score GROUP BY date;
