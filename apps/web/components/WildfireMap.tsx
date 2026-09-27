"use client";

import { useEffect, useRef, useCallback } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { View } from "./ViewSwitcher";
import { SelectedFire } from "@/app/page";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// BC centre
const BC_CENTER: [number, number] = [-125.5, 54.0];
const BC_ZOOM = 5;

// FIRMS WMS tile URL (public, no auth — visual fallback when no MAP_KEY)
const FIRMS_WMS_TILES =
  "https://firms.modaps.eosdis.nasa.gov/mapserver/wms/fires/?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=fires_viirs_snpp&STYLES=&FORMAT=image/png&TRANSPARENT=true&CRS=EPSG:3857&WIDTH=256&HEIGHT=256&BBOX={bbox-epsg-3857}";

interface Props {
  view: View;
  historyYear: number;
  onFireSelect: (fire: SelectedFire) => void;
  selectedFire: SelectedFire | null;
}

export default function WildfireMap({ view, historyYear, onFireSelect, selectedFire }: Props) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const popup = useRef<maplibregl.Popup | null>(null);

  // ── Bootstrap map ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          "esri-satellite": {
            type: "raster",
            tiles: [
              "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            ],
            tileSize: 256,
            attribution: "© Esri, Maxar, GeoEye, Earthstar Geographics",
          },
          "osm-labels": {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [
          { id: "esri-satellite", type: "raster", source: "esri-satellite", maxzoom: 19 },
          { id: "osm-labels",     type: "raster", source: "osm-labels",     maxzoom: 19, paint: { "raster-opacity": 0.35 } },
        ],
      },
      center: BC_CENTER,
      zoom: BC_ZOOM,
    });

    map.current.addControl(new maplibregl.NavigationControl(), "bottom-right");
    map.current.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");

    popup.current = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      className: "wildfire-popup",
    });

    return () => {
      map.current?.remove();
      map.current = null;
    };
  }, []);

  // ── Layer management ───────────────────────────────────────────────────────
  const removeLayer = useCallback((id: string) => {
    if (!map.current) return;
    if (map.current.getLayer(id)) map.current.removeLayer(id);
    if (map.current.getSource(id)) map.current.removeSource(id);
  }, []);

  const clearAllDataLayers = useCallback(() => {
    const layers = [
      "active-fires", "active-fires-glow",
      "hotspots", "firms-wms",
      "history-fill", "history-outline",
      "spread-p75", "spread-p50", "spread-p25",
      "risk-heat",
      "fwi-stations",
    ];
    layers.forEach(removeLayer);
  }, [removeLayer]);

  // ── View: Current (active fires + FIRMS hotspots + FWI) ───────────────────
  const loadCurrentView = useCallback(async () => {
    if (!map.current) return;
    clearAllDataLayers();

    // 1. Active BC fires
    try {
      const res = await fetch(`${API}/api/v1/fires/active`);
      const data = await res.json();
      if (data.features?.length > 0) {
        map.current.addSource("active-fires", { type: "geojson", data });
        map.current.addLayer({
          id: "active-fires-glow",
          type: "circle",
          source: "active-fires",
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["get", "size_ha"], 0, 8, 1000, 16, 10000, 28],
            "circle-color": "#f97316",
            "circle-opacity": 0.25,
            "circle-blur": 1,
          },
        });
        map.current.addLayer({
          id: "active-fires",
          type: "circle",
          source: "active-fires",
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["get", "size_ha"], 0, 5, 1000, 10, 10000, 18],
            "circle-color": "#f97316",
            "circle-opacity": 0.9,
            "circle-stroke-color": "#fff",
            "circle-stroke-width": 1,
          },
        });

        map.current.on("click", "active-fires", (e) => {
          const f = e.features?.[0];
          if (!f || !f.geometry || f.geometry.type !== "Point") return;
          const props = f.properties as Record<string, unknown>;
          onFireSelect({
            fire_id: String(props.fire_number || "unknown"),
            fire_name: String(props.fire_name || props.fire_number || ""),
            lat: (f.geometry as GeoJSON.Point).coordinates[1],
            lon: (f.geometry as GeoJSON.Point).coordinates[0],
            size_ha: Number(props.size_ha) || 0,
            source: "active",
            properties: props,
          });
        });
        map.current.on("mouseenter", "active-fires", () => {
          if (map.current) map.current.getCanvas().style.cursor = "pointer";
        });
        map.current.on("mouseleave", "active-fires", () => {
          if (map.current) map.current.getCanvas().style.cursor = "";
        });
      }
    } catch {}

    // 2. FIRMS hotspots (point data if MAP_KEY available, else WMS tiles)
    try {
      const res = await fetch(`${API}/api/v1/fires/hotspots?days=1`);
      const data = await res.json();
      if (data.features?.length > 0) {
        map.current.addSource("hotspots", { type: "geojson", data });
        map.current.addLayer({
          id: "hotspots",
          type: "circle",
          source: "hotspots",
          paint: {
            "circle-radius": 4,
            "circle-color": ["interpolate", ["linear"], ["get", "frp"], 0, "#fbbf24", 50, "#f97316", 200, "#dc2626"],
            "circle-opacity": 0.85,
          },
        });
      } else {
        // WMS fallback
        map.current.addSource("firms-wms", {
          type: "raster",
          tiles: [FIRMS_WMS_TILES],
          tileSize: 256,
        });
        map.current.addLayer({ id: "firms-wms", type: "raster", source: "firms-wms", paint: { "raster-opacity": 0.8 } });
      }
    } catch {}

    // 3. FWI stations
    try {
      const res = await fetch(`${API}/api/v1/weather/fwi`);
      const data = await res.json();
      if (data.features?.length > 0) {
        map.current.addSource("fwi-stations", { type: "geojson", data });
        map.current.addLayer({
          id: "fwi-stations",
          type: "circle",
          source: "fwi-stations",
          paint: {
            "circle-radius": 6,
            "circle-color": [
              "step", ["get", "FWI"],
              "#22c55e", 5,
              "#eab308", 12,
              "#f97316", 20,
              "#ef4444", 30,
              "#7c3aed",
            ],
            "circle-opacity": 0.7,
            "circle-stroke-color": "#fff",
            "circle-stroke-width": 0.5,
          },
        });
      }
    } catch {}
  }, [clearAllDataLayers, onFireSelect]);

  // ── View: History ──────────────────────────────────────────────────────────
  const loadHistoryView = useCallback(async (year: number) => {
    if (!map.current) return;
    clearAllDataLayers();

    try {
      const res = await fetch(`${API}/api/v1/fires/history?year=${year}`);
      const data = await res.json();
      if (!data.features?.length) return;

      map.current.addSource("history-fill", { type: "geojson", data });
      map.current.addLayer({
        id: "history-fill",
        type: "fill",
        source: "history-fill",
        paint: {
          "fill-color": [
            "interpolate", ["linear"], ["get", "size_ha"],
            0, "#fef9c3", 100, "#fde047", 1000, "#f97316", 10000, "#dc2626", 100000, "#7c3aed",
          ],
          "fill-opacity": 0.55,
        },
      });
      map.current.addLayer({
        id: "history-outline",
        type: "line",
        source: "history-fill",
        paint: { "line-color": "#f97316", "line-width": 0.5, "line-opacity": 0.8 },
      });

      map.current.on("click", "history-fill", (e) => {
        const f = e.features?.[0];
        if (!f) return;
        const props = f.properties as Record<string, unknown>;
        const centroid = e.lngLat;
        onFireSelect({
          fire_id: String(props.fire_number || ""),
          fire_name: String(props.fire_number || ""),
          lat: centroid.lat,
          lon: centroid.lng,
          size_ha: Number(props.size_ha) || 0,
          source: "history",
          properties: props,
        });
      });
    } catch {}
  }, [clearAllDataLayers, onFireSelect]);

  // ── View: Predictions ─────────────────────────────────────────────────────
  const loadPredictionsView = useCallback(async () => {
    if (!map.current) return;
    clearAllDataLayers();

    // Load active fires first, then show spread forecast on click
    try {
      const res = await fetch(`${API}/api/v1/fires/active`);
      const data = await res.json();
      if (!data.features?.length) return;

      map.current.addSource("active-fires", { type: "geojson", data });
      map.current.addLayer({
        id: "active-fires",
        type: "circle",
        source: "active-fires",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "size_ha"], 0, 6, 10000, 14],
          "circle-color": "#f97316",
          "circle-opacity": 0.9,
          "circle-stroke-color": "#fff",
          "circle-stroke-width": 1.5,
        },
      });

      map.current.on("click", "active-fires", async (e) => {
        const f = e.features?.[0];
        if (!f || !f.geometry || f.geometry.type !== "Point") return;
        const [lon, lat] = (f.geometry as GeoJSON.Point).coordinates;
        const props = f.properties as Record<string, unknown>;

        onFireSelect({
          fire_id: String(props.fire_number || ""),
          fire_name: String(props.fire_name || ""),
          lat, lon,
          size_ha: Number(props.size_ha) || 0,
          source: "active",
          properties: props,
        });

        // Fetch and show spread forecast
        const fwi = 22, isi = 9, windSpeed = 5, windDir = 270;
        try {
          const sres = await fetch(
            `${API}/api/v1/predict/spread?lat=${lat}&lon=${lon}&fwi=${fwi}&isi=${isi}&wind_speed=${windSpeed}&wind_dir=${windDir}&hours=24`
          );
          const spread = await sres.json();
          addSpreadLayers(spread);
        } catch {}
      });
    } catch {}
  }, [clearAllDataLayers, onFireSelect]);

  function addSpreadLayers(spread: GeoJSON.FeatureCollection) {
    if (!map.current) return;
    ["spread-p75", "spread-p50", "spread-p25"].forEach((id) => {
      if (map.current!.getLayer(id)) map.current!.removeLayer(id);
      if (map.current!.getSource(id)) map.current!.removeSource(id);
    });

    const byPercentile: Record<string, GeoJSON.Feature> = {};
    for (const f of (spread as GeoJSON.FeatureCollection).features) {
      const p = (f.properties as Record<string, string>)?.percentile;
      if (p) byPercentile[p] = f;
    }

    const configs = [
      { id: "spread-p75", key: "p75", color: "#fca5a5", opacity: 0.25 },
      { id: "spread-p50", key: "p50", color: "#f97316", opacity: 0.40 },
      { id: "spread-p25", key: "p25", color: "#ef4444", opacity: 0.55 },
    ];

    for (const { id, key, color, opacity } of configs) {
      const feat = byPercentile[key];
      if (!feat) continue;
      map.current.addSource(id, { type: "geojson", data: { type: "FeatureCollection", features: [feat] } });
      map.current.addLayer({
        id,
        type: "fill",
        source: id,
        paint: { "fill-color": color, "fill-opacity": opacity },
      });
    }
  }

  // ── View: Risk ─────────────────────────────────────────────────────────────
  const loadRiskView = useCallback(async () => {
    if (!map.current) return;
    clearAllDataLayers();

    try {
      const res = await fetch(`${API}/api/v1/risk/map`);
      const data = await res.json();
      if (!data.features?.length) return;

      map.current.addSource("risk-heat", { type: "geojson", data });
      map.current.addLayer({
        id: "risk-heat",
        type: "heatmap",
        source: "risk-heat",
        paint: {
          "heatmap-weight": ["interpolate", ["linear"], ["get", "burn_probability"], 0, 0, 1, 1],
          "heatmap-intensity": 1.2,
          "heatmap-radius": 30,
          "heatmap-color": [
            "interpolate", ["linear"], ["heatmap-density"],
            0,   "rgba(0,0,0,0)",
            0.2, "#22c55e",
            0.4, "#eab308",
            0.6, "#f97316",
            0.8, "#ef4444",
            1,   "#7c3aed",
          ],
          "heatmap-opacity": 0.75,
        },
      });
    } catch {}
  }, [clearAllDataLayers]);

  // ── React to view changes ──────────────────────────────────────────────────
  useEffect(() => {
    if (!map.current) return;
    const onLoad = () => {
      if (view === "current")     loadCurrentView();
      if (view === "history")     loadHistoryView(historyYear);
      if (view === "predictions") loadPredictionsView();
      if (view === "risk")        loadRiskView();
    };

    if (map.current.isStyleLoaded()) onLoad();
    else map.current.once("load", onLoad);
  }, [view, historyYear, loadCurrentView, loadHistoryView, loadPredictionsView, loadRiskView]);

  // ── Fly to selected fire ───────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedFire || !map.current) return;
    map.current.flyTo({
      center: [selectedFire.lon, selectedFire.lat],
      zoom: Math.max(map.current.getZoom(), 8),
      speed: 1.2,
    });
  }, [selectedFire]);

  return (
    <div ref={mapContainer} className="w-full h-full" />
  );
}
