"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import ComingSoon from "@/components/ComingSoon";
import Panel, { type ReportState } from "@/components/Panel";
import ViewSwitcher from "@/components/ViewSwitcher";
import { ApiError, apiBase, getHealth, getHotspots, getWeather, postReport } from "@/lib/api";
import type { HealthResponse, HotspotsResponse, WeatherResponse } from "@/lib/types";
import { DEFAULT_VIEW, activeLayers, viewById, type ViewId } from "@/lib/views";

function MapFallback() {
  return <div className="map-fallback">Loading map…</div>;
}

const FireMap = dynamic(() => import("@/components/FireMap"), {
  ssr: false,
  loading: () => <MapFallback />,
});

type HotspotStatus = "loading" | "ready" | "missing-key" | "error";

export default function HomePage() {
  const requestId = useRef(0);
  const reportRequest = useRef(0);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [hotspots, setHotspots] = useState<HotspotsResponse | null>(null);
  const [hotspotStatus, setHotspotStatus] = useState<HotspotStatus>("loading");
  const [hotspotMessage, setHotspotMessage] = useState<string | null>(null);
  const [weather, setWeather] = useState<WeatherResponse | null>(null);
  const [weatherNote, setWeatherNote] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showStations, setShowStations] = useState(false);
  const [report, setReport] = useState<ReportState | null>(null);
  const [viewId, setViewId] = useState<ViewId>(DEFAULT_VIEW);
  const view = viewById(viewId);
  const layers = activeLayers(view);

  const load = useCallback(() => {
    const id = requestId.current + 1;
    requestId.current = id;
    setHotspotStatus("loading");
    setHotspotMessage(null);
    setHealth(null);
    setWeather(null);
    setWeatherNote(null);

    getHealth()
      .then((data) => {
        if (requestId.current === id) setHealth(data);
      })
      .catch(() => {
        if (requestId.current === id) setHealth(null);
      });

    getHotspots()
      .then((data) => {
        if (requestId.current !== id) return;
        setHotspots(data);
        setHotspotStatus("ready");
        setSelectedId((current) => {
          if (current && data.clusters.some((cluster) => cluster.id === current)) return current;
          return null;
        });
      })
      .catch((error: unknown) => {
        if (requestId.current !== id) return;
        const message =
          error instanceof ApiError
            ? error.message
            : `Cannot reach the API at ${apiBase()}. Start the API and set NEXT_PUBLIC_API_URL.`;
        const missingKey =
          error instanceof ApiError && error.status === 503 && message.includes("FIRMS_MAP_KEY");
        setHotspots(null);
        setHotspotStatus(missingKey ? "missing-key" : "error");
        setHotspotMessage(message);
        setSelectedId(null);
      });

    getWeather()
      .then((data) => {
        if (requestId.current === id) setWeather(data);
      })
      .catch(() => {
        if (requestId.current === id) {
          setWeatherNote("Fire weather could not be loaded from the API.");
        }
      });
  }, []);

  useEffect(() => {
    load();
    return () => {
      requestId.current += 1;
    };
  }, [load]);

  const selected = hotspots?.clusters.find((cluster) => cluster.id === selectedId) ?? null;

  async function onGenerate() {
    if (!selected) return;
    const cluster = selected;
    const token = reportRequest.current + 1;
    reportRequest.current = token;
    setReport({ clusterId: cluster.id, loading: true });
    try {
      const result = await postReport(cluster);
      if (reportRequest.current !== token) return;
      setReport({ clusterId: cluster.id, loading: false, text: result.report, result });
    } catch (error: unknown) {
      if (reportRequest.current !== token) return;
      const message =
        error instanceof ApiError ? error.message : "Could not generate a situation report.";
      setReport({ clusterId: cluster.id, loading: false, error: message });
    }
  }

  const overlay =
    view.available && (hotspotStatus === "missing-key" || hotspotStatus === "error") ? (
      <div className={`map-overlay ${hotspotStatus === "error" ? "is-error" : ""}`} role="status">
        <p>{hotspotStatus === "missing-key" ? "Waiting on a FIRMS map key" : "Map data unavailable"}</p>
      </div>
    ) : null;

  return (
    <div className="app">
      <a className="skip" href="#map-panel">
        Skip to map panel
      </a>
      <header className="topbar">
        <div className="brand">
          <span className="ember" aria-hidden="true" />
          <div>
            <p className="wordmark">BC Wildfire Watch</p>
            <p className="tag">{view.summary}</p>
          </div>
        </div>
        <div className="chips">
          <span className={health ? "chip ok" : "chip"}>
            API {health ? "connected" : hotspotStatus === "loading" ? "checking" : "unreachable"}
          </span>
          <span className={health?.firms_configured ? "chip ok" : health ? "chip warn" : "chip"}>
            FIRMS {health ? (health.firms_configured ? "key set" : "key missing") : "checking"}
          </span>
          <span className={health?.grok_configured ? "chip ok" : health ? "chip warn" : "chip"}>
            Grok {health ? (health.grok_configured ? "key set" : "key missing") : "checking"}
          </span>
          <button type="button" className="reload" onClick={load}>
            Reload
          </button>
        </div>
      </header>
      <ViewSwitcher active={viewId} onChange={setViewId} />
      <div className="workspace">
        <div className="map-wrap">
          <FireMap
            activeLayers={layers}
            clusters={hotspots?.clusters ?? []}
            stations={weather?.stations ?? []}
            showStations={showStations}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
          {overlay}
        </div>
        {view.available ? (
          <Panel
            hotspotStatus={hotspotStatus}
            hotspotMessage={hotspotMessage}
            hotspots={hotspots}
            weather={weather}
            weatherNote={weatherNote}
            health={health}
            selectedId={selectedId}
            onSelect={setSelectedId}
            showStations={showStations}
            onToggleStations={setShowStations}
            report={report}
            onGenerate={() => {
              void onGenerate();
            }}
          />
        ) : (
          <ComingSoon view={view} />
        )}
      </div>
    </div>
  );
}
