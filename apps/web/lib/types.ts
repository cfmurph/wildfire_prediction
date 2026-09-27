export type HealthResponse = {
  status: string;
  region: string;
  firms_configured: boolean;
  grok_configured: boolean;
};

export type HotspotCluster = {
  id: string;
  latitude: number;
  longitude: number;
  hotspot_count: number;
  max_frp: number | null;
  mean_frp: number | null;
  max_brightness: number | null;
  latest_acquisition: string | null;
  confidence: Record<string, number>;
  satellites: string[];
  daynight: Record<string, number>;
  bbox: number[];
};

export type HotspotsResponse = {
  region: string;
  bbox: { west: number; south: number; east: number; north: number };
  sources: string[];
  day_range: number;
  fetched_at: string;
  cached: boolean;
  hotspot_count: number;
  cluster_count: number;
  clusters: HotspotCluster[];
  warnings: string[];
};

export type WeatherStation = {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  observed_at: string;
  temp_c: number | null;
  relative_humidity: number | null;
  wind_speed_kmh: number | null;
  wind_direction_deg: number | null;
  precip_mm: number | null;
  ffmc: number | null;
  dmc: number | null;
  dc: number | null;
  bui: number | null;
  isi: number | null;
  fwi: number | null;
};

export type WeatherResponse = {
  source: string;
  available: boolean;
  detail: string | null;
  fetched_at: string;
  cached: boolean;
  station_count: number;
  stations: WeatherStation[];
};

export type ReportResponse = {
  report: string;
  model: string;
  ml_forecast_available: boolean;
  approximate_area_ha: number;
  fire_weather: {
    id: string;
    name: string;
    distance_km: number;
    fwi: number | null;
    isi: number | null;
    bui: number | null;
    ffmc: number | null;
    wind_speed_kmh: number | null;
    wind_direction_deg: number | null;
    observed_at: string | null;
  } | null;
  disclaimer: string;
  inputs: {
    fire_id: string;
    as_of: string;
    approximate_area_ha: number;
    fwi: number;
    isi: number;
    bui: number;
    ffmc: number;
    wind_speed_kmh: number;
    wind_direction: string;
    station_name: string | null;
  };
};
