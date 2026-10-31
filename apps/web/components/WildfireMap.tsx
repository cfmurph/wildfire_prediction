"use client";

import { useEffect, useRef, useCallback } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { View } from "./ViewSwitcher";
import { SelectedFire } from "@/app/page";
import { FIRE_STATUSES } from "@/lib/utils";
import MapLegend from "./MapLegend";
import { activeFire, CurrentData } from "@/lib/fires";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// BC centre
const BC_CENTER: [number, number] = [-125.5, 54.0];
const BC_ZOOM = 5;

// FIRMS WMS tile URL (public, no auth — visual fallback when no MAP_KEY)
const FIRMS_WMS_TILES =
  "https://firms.modaps.eosdis.nasa.gov/mapserver/wms/fires/?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&LAYERS=fires_viirs_snpp&STYLES=&FORMAT=image/png&TRANSPARENT=true&CRS=EPSG:3857&WIDTH=256&HEIGHT=256&BBOX={bbox-epsg-3857}";

type LayerEvent = "click" | "mouseenter" | "mouseleave";

// Fire circles coloured by BCWS stage of control; grey if unknown
const STATUS_COLOR: maplibregl.ExpressionSpecification = [
  "match", ["get", "fire_status"],
  ...FIRE_STATUSES.flatMap((s) => [s.status, s.color]),
  "#9ca3af",
] as unknown as maplibregl.ExpressionSpecification;

interface Props {
  view: View;
  onCurrentData: (data: CurrentData) => void;
  historyYear: number;
  riskMonth: number;
  riskLayers: { lightning: boolean; human: boolean };
  onFireSelect: (fire: SelectedFire) => void;
  selectedFire: SelectedFire | null;
  onRiskPointClick: (point: { lat: number; lon: number }) => void;
}

export default function WildfireMap({ view, onCurrentData, historyYear, riskMonth, riskLayers, onFireSelect, selectedFire, onRiskPointClick }: Props) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const popup = useRef<maplibregl.Popup | null>(null);
  // Layer event handlers registered by the current view; removed on view change so they don't pile up
  const handlers = useRef<[LayerEvent, string, (e: maplibregl.MapLayerMouseEvent) => void][]>([]);

  const on = useCallback((type: LayerEvent, layer: string, fn: (e: maplibregl.MapLayerMouseEvent) => void) => {
    map.current?.on(type, layer, fn);
    handlers.current.push([type, layer, fn]);
  }, []);

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
  }, []);

  const removeSource = useCallback((id: string) => {
    if (!map.current) return;
    if (map.current.getSource(id)) map.current.removeSource(id);
  }, []);

  const clearAllDataLayers = useCallback(() => {
    // Remove all layers first, then sources (a source can't be removed while layers use it)
    const layers = [
      "active-fires-glow", "active-fires",
      "hotspots", "firms-wms",
      "history-fill", "history-outline",
      "spread-p75", "spread-p50", "spread-p25",
      "risk-heat",
      "ignition-lightning", "ignition-human",
      "fwi-stations",
    ];
    handlers.current.forEach(([type, layer, fn]) => map.current?.off(type, layer, fn));
    handlers.current = [];
    layers.forEach(removeLayer);
    // Now safe to remove sources
    const sources = [
      "active-fires", "hotspots", "firms-wms",
      "history-fill", "spread-p75", "spread-p50", "spread-p25",
      "risk-heat", "ignition-lightning", "ignition-human", "fwi-stations",
    ];
    sources.forEach(removeSource);
  }, [removeLayer, removeSource]);

  // ── View: Current (active fires + FIRMS hotspots + FWI) ───────────────────
  const loadCurrentView = useCallback(async () => {
    if (!map.current) return;
    clearAllDataLayers();
    const getJson = (path: string) => fetch(`${API}${path}`).then((r) => r.json()).catch(() => null);
    const [fires, hotspots, stations] = await Promise.all([
      getJson("/api/v1/fires/active"),
      getJson("/api/v1/fires/hotspots?days=1"),
      getJson("/api/v1/weather/fwi"),
    ]);
    if (!map.current) return;
    onCurrentData({
      fires: fires?.features ?? [],
      hotspots: hotspots?.features ?? [],
      stations: stations?.features?.length ?? 0,
      updated: new Date(),
    });

    // Satellite hotspots go underneath the fires: small pale dots
    if (hotspots?.features?.length) {
      map.current.addSource("hotspots", { type: "geojson", data: hotspots });
      map.current.addLayer({
        id: "hotspots",
        type: "circle",
        source: "hotspots",
        paint: { "circle-radius": 2.5, "circle-color": "#fde68a", "circle-opacity": 0.8 },
      });
    } else if (hotspots) {
      // No FIRMS key: public WMS tiles instead of points
      map.current.addSource("firms-wms", { type: "raster", tiles: [FIRMS_WMS_TILES], tileSize: 256 });
      map.current.addLayer({ id: "firms-wms", type: "raster", source: "firms-wms", paint: { "raster-opacity": 0.8 } });
    }

    // Weather stations: squares would need a sprite, so hollow rings coloured by FWI class
    if (stations?.features?.length) {
      map.current.addSource("fwi-stations", { type: "geojson", data: stations });
      map.current.addLayer({
        id: "fwi-stations",
        type: "circle",
        source: "fwi-stations",
        paint: {
          "circle-radius": 5,
          "circle-color": "rgba(0,0,0,0)",
          "circle-stroke-width": 2,
          "circle-stroke-color": [
            "step", ["get", "FWI"],
            "#22c55e", 5, "#eab308", 12, "#f97316", 20, "#ef4444", 30, "#7c3aed",
          ],
        },
      });
    }

    if (fires?.features?.length) {
      map.current.addSource("active-fires", { type: "geojson", data: fires });
      map.current.addLayer({
        id: "active-fires",
        type: "circle",
        source: "active-fires",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "size_ha"], 0, 5, 1000, 9, 10000, 15, 100000, 24],
          "circle-color": STATUS_COLOR,
          "circle-opacity": 0.85,
          "circle-stroke-color": "#111827",
          "circle-stroke-width": 1.5,
        },
      });
      on("click", "active-fires", (e) => {
        const f = e.features?.[0];
        if (f) onFireSelect(activeFire(f));
      });
      on("mouseenter", "active-fires", () => { if (map.current) map.current.getCanvas().style.cursor = "pointer"; });
      on("mouseleave", "active-fires", () => { if (map.current) map.current.getCanvas().style.cursor = ""; });
    }
  }, [clearAllDataLayers, on, onFireSelect, onCurrentData]);

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

      on("click", "history-fill", (e) => {
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
  }, [clearAllDataLayers, on, onFireSelect]);

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
          "circle-radius": ["interpolate", ["linear"], ["get", "size_ha"], 0, 5, 1000, 9, 10000, 15, 100000, 24],
          "circle-color": STATUS_COLOR,
          "circle-opacity": 0.85,
          "circle-stroke-color": "#111827",
          "circle-stroke-width": 1.5,
        },
      });

      on("click", "active-fires", async (e) => {
        const f = e.features?.[0];
        if (!f) return;
        const fire = activeFire(f);
        const { lat, lon } = fire;
        onFireSelect(fire);

        // No spread estimate without real weather
        let stn: Record<string, number> | null = null;
        try {
          stn = await (await fetch(`${API}/api/v1/weather/nearest?lat=${lat}&lon=${lon}`)).json();
        } catch {}
        if (!stn) return;
        try {
          const sres = await fetch(
            `${API}/api/v1/predict/spread?lat=${lat}&lon=${lon}&fwi=${stn.FWI}&isi=${stn.ISI}&wind_speed=${stn.wind_speed / 3.6}&wind_dir=${stn.wind_dir}&hours=24`
          );
          const spread = await sres.json();
          addSpreadLayers(spread);
        } catch {}
      });
    } catch {}
  }, [clearAllDataLayers, on, onFireSelect]);

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
  const loadRiskView = useCallback(async (month: number) => {
    if (!map.current) return;
    clearAllDataLayers();

    // Load climatology risk surface AND ignition prediction in parallel
    const [riskRes, ignitionRes] = await Promise.allSettled([
      fetch(`${API}/api/v1/risk/monthly?month=${month}`),
      fetch(`${API}/api/v1/ignition/causes?month=${month}`),
    ]);

    // Climatology layer (background context)
    if (riskRes.status === "fulfilled" && riskRes.value.ok) {
      try {
        const data = await riskRes.value.json();
        if (data.features?.length) {
          map.current.addSource("risk-heat", { type: "geojson", data });
          map.current.addLayer({
            id: "risk-heat",
            type: "circle",
            source: "risk-heat",
            paint: {
              "circle-radius": 20,
              "circle-blur": 1.5,
              "circle-opacity": 0.35,
              "circle-color": [
                "step", ["get", "burn_probability"],
                "#22c55e", 0.03, "#eab308", 0.08, "#f97316", 0.18, "#ef4444",
              ],
            },
          });
        }
      } catch {}
    }

    // Ignition prediction layer — lightning (blue) and human (orange) separately
    if (ignitionRes.status === "fulfilled" && ignitionRes.value.ok) {
      try {
        const causes = await ignitionRes.value.json();

        // Lightning-caused ignition risk (blue tones) — C: clickable for spread forecast
        const lightning = causes.lightning;
        if (lightning?.features?.length) {
          map.current.addSource("ignition-lightning", { type: "geojson", data: lightning });
          map.current.addLayer({
            id: "ignition-lightning",
            type: "circle",
            source: "ignition-lightning",
            paint: {
              "circle-radius": ["interpolate", ["linear"], ["get", "prob"], 0, 8, 0.5, 22],
              "circle-color": "#3b82f6",
              "circle-opacity": 0.75,
              "circle-stroke-color": "#1d4ed8",
              "circle-stroke-width": 0.5,
              "circle-blur": 0.3,
            },
          });
          on("click", "ignition-lightning", (e) => {
            const f = e.features?.[0];
            if (!f?.geometry || f.geometry.type !== "Point") return;
            const [lon, lat] = (f.geometry as GeoJSON.Point).coordinates;
            onRiskPointClick({ lat, lon });
          });
          on("mouseenter", "ignition-lightning", () => {
            if (map.current) map.current.getCanvas().style.cursor = "crosshair";
          });
          on("mouseleave", "ignition-lightning", () => {
            if (map.current) map.current.getCanvas().style.cursor = "";
          });
        }

        // Human-caused ignition risk (amber tones) — also clickable
        const human = causes.human;
        if (human?.features?.length) {
          map.current.addSource("ignition-human", { type: "geojson", data: human });
          map.current.addLayer({
            id: "ignition-human",
            type: "circle",
            source: "ignition-human",
            paint: {
              "circle-radius": ["interpolate", ["linear"], ["get", "prob"], 0, 7, 0.5, 18],
              "circle-color": "#f59e0b",
              "circle-opacity": 0.75,
              "circle-stroke-color": "#d97706",
              "circle-stroke-width": 0.5,
              "circle-blur": 0.3,
            },
          });
          on("click", "ignition-human", (e) => {
            const f = e.features?.[0];
            if (!f?.geometry || f.geometry.type !== "Point") return;
            const [lon, lat] = (f.geometry as GeoJSON.Point).coordinates;
            onRiskPointClick({ lat, lon });
          });
          on("mouseenter", "ignition-human", () => {
            if (map.current) map.current.getCanvas().style.cursor = "crosshair";
          });
          on("mouseleave", "ignition-human", () => {
            if (map.current) map.current.getCanvas().style.cursor = "";
          });
        }
      } catch {}
    }
  }, [clearAllDataLayers, on, onRiskPointClick]);

  // ── React to view changes ──────────────────────────────────────────────────
  useEffect(() => {
    if (!map.current) return;
    const onLoad = () => {
      if (view === "current")     loadCurrentView();
      if (view === "history")     loadHistoryView(historyYear);
      if (view === "predictions") loadPredictionsView();
      if (view === "risk")        loadRiskView(riskMonth);
    };

    if (map.current.isStyleLoaded()) onLoad();
    else map.current.once("load", onLoad);
  }, [view, historyYear, riskMonth, loadCurrentView, loadHistoryView, loadPredictionsView, loadRiskView]);

  // ── React to layer toggle changes (B) ─────────────────────────────────────
  useEffect(() => {
    if (!map.current || view !== "risk") return;
    const setVis = (id: string, visible: boolean) => {
      if (map.current?.getLayer(id)) {
        map.current.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
      }
    };
    setVis("ignition-lightning", riskLayers.lightning);
    setVis("ignition-human", riskLayers.human);
  }, [riskLayers, view]);

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
    <div className="relative w-full h-full">
      <div ref={mapContainer} className="w-full h-full" />
      {(view === "current" || view === "predictions") && <MapLegend showSpread={view === "predictions"} />}
    </div>
  );
}
