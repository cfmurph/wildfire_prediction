"use client";

import { useMemo } from "react";
import type { HealthResponse, HotspotCluster, HotspotsResponse, ReportResponse, WeatherResponse } from "@/lib/types";
import {
  formatCoord,
  formatFrp,
  formatKm,
  formatNumber,
  formatUtc,
  frpColor,
  fwiBand,
  fwiColor,
  nearestStation,
  satelliteLabel,
} from "@/lib/format";

export type ReportState = {
  clusterId: string;
  loading: boolean;
  text?: string;
  error?: string;
  result?: ReportResponse;
};

type PanelProps = {
  hotspotStatus: "loading" | "ready" | "missing-key" | "error";
  hotspotMessage: string | null;
  hotspots: HotspotsResponse | null;
  weather: WeatherResponse | null;
  weatherNote: string | null;
  health: HealthResponse | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  showStations: boolean;
  onToggleStations: (value: boolean) => void;
  report: ReportState | null;
  onGenerate: () => void;
};

export default function Panel({
  hotspotStatus,
  hotspotMessage,
  hotspots,
  weather,
  weatherNote,
  health,
  selectedId,
  onSelect,
  showStations,
  onToggleStations,
  report,
  onGenerate,
}: PanelProps) {
  const clusters = hotspots?.clusters ?? [];
  const selected = clusters.find((cluster) => cluster.id === selectedId) ?? null;
  const activeReport = report && selected && report.clusterId === selected.id ? report : null;
  const nearest = useMemo(() => {
    if (!selected || !weather?.stations.length) return null;
    return nearestStation(selected.latitude, selected.longitude, weather.stations);
  }, [selected, weather]);

  const grokKnown = health != null;
  const grokReady = health?.grok_configured === true;
  const reportDisabled = !selected || activeReport?.loading === true || (grokKnown && !grokReady);

  return (
    <aside className="panel" id="clusters">
      <div className="panel-block">
        <p className="eyebrow">British Columbia</p>
        <h2>Hotspot clusters</h2>
        <p className="lede">
          VIIRS detections are grouped into about 15–20 km cells. Select one to read fire weather
          and generate a situation report.
        </p>
        <Legend />
      </div>

      {hotspotStatus === "loading" ? <Skeleton /> : null}

      {hotspotStatus === "missing-key" ? (
        <div className="notice notice-warn" role="status">
          <strong>NASA FIRMS key required.</strong>
          <p>{hotspotMessage}</p>
        </div>
      ) : null}

      {hotspotStatus === "error" ? (
        <div className="notice notice-error" role="alert">
          <strong>Hotspots unavailable.</strong>
          <p>{hotspotMessage}</p>
        </div>
      ) : null}

      {hotspotStatus === "ready" && hotspots ? (
        <>
          <div className="panel-block meta-row">
            <span>{hotspots.hotspot_count} detections</span>
            <span>{hotspots.cluster_count} clusters</span>
            <span>last {hotspots.day_range}d</span>
            <span>{hotspots.cached ? "cached" : "fresh"}</span>
          </div>
          {hotspots.warnings.length > 0 ? (
            <p className="warn-line">{hotspots.warnings.join(" ")}</p>
          ) : null}
          {hotspots.cluster_count === 0 ? (
            <div className="notice" role="status">
              <strong>No active VIIRS hotspots in this window.</strong>
              <p>
                British Columbia can be quiet, especially outside peak fire season. This view
                covers the last {hotspots.day_range} days inside the provincial bounding box.
              </p>
            </div>
          ) : null}

          {selected ? (
            <Detail
              cluster={selected}
              nearest={nearest}
              weather={weather}
              weatherNote={weatherNote}
              showStations={showStations}
              onToggleStations={onToggleStations}
              grokKnown={grokKnown}
              grokReady={grokReady}
              reportDisabled={reportDisabled}
              activeReport={activeReport}
              onGenerate={onGenerate}
            />
          ) : (
            <div className="panel-block">
              <p className="hint">Select a cluster on the map or from the list.</p>
              <StationToggle
                showStations={showStations}
                onToggleStations={onToggleStations}
                disabled={!weather?.stations.length}
              />
            </div>
          )}

          <ul className="cluster-list">
            {clusters.map((cluster) => (
              <li key={cluster.id}>
                <button
                  type="button"
                  className="cluster-button"
                  aria-pressed={cluster.id === selectedId}
                  onClick={() => onSelect(cluster.id)}
                >
                  <span className="swatch" style={{ background: frpColor(cluster.max_frp) }} />
                  <span className="cluster-copy">
                    <span className="cluster-title">{formatCoord(cluster.latitude, cluster.longitude)}</span>
                    <span className="cluster-sub">
                      {cluster.hotspot_count} detections · {formatFrp(cluster.max_frp)} ·{" "}
                      {formatUtc(cluster.latest_acquisition)}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      <p className="disclaimer">
        Research prototype. Not an official BC Wildfire Service product or evacuation order.
        Sources: NASA FIRMS VIIRS near-real-time and CWFIS fire-weather stations. Next-day spread
        is not forecast in this view.
      </p>
    </aside>
  );
}

function Legend() {
  return (
    <ul className="legend">
      <li><i style={{ background: "#e6c07b" }} /> FRP under 5 MW</li>
      <li><i style={{ background: "#f08a24" }} /> 5–20 MW</li>
      <li><i style={{ background: "#e23d2b" }} /> 20 MW and above</li>
    </ul>
  );
}

function Skeleton() {
  return (
    <div className="skeleton-wrap" aria-hidden="true">
      <div className="skeleton" />
      <div className="skeleton" />
      <div className="skeleton short" />
    </div>
  );
}

function StationToggle({
  showStations,
  onToggleStations,
  disabled,
}: {
  showStations: boolean;
  onToggleStations: (value: boolean) => void;
  disabled: boolean;
}) {
  return (
    <label className="toggle">
      <input
        type="checkbox"
        checked={showStations}
        disabled={disabled}
        onChange={(event) => onToggleStations(event.target.checked)}
      />
      Show CWFIS weather stations
    </label>
  );
}

function Detail({
  cluster,
  nearest,
  weather,
  weatherNote,
  showStations,
  onToggleStations,
  grokKnown,
  grokReady,
  reportDisabled,
  activeReport,
  onGenerate,
}: {
  cluster: HotspotCluster;
  nearest: { station: WeatherResponse["stations"][number]; distanceKm: number } | null;
  weather: WeatherResponse | null;
  weatherNote: string | null;
  showStations: boolean;
  onToggleStations: (value: boolean) => void;
  grokKnown: boolean;
  grokReady: boolean;
  reportDisabled: boolean;
  activeReport: ReportState | null;
  onGenerate: () => void;
}) {
  const day = cluster.daynight.D ?? 0;
  const night = cluster.daynight.N ?? 0;
  const confidence = ["high", "nominal", "low"]
    .map((key) => `${cluster.confidence[key] ?? 0} ${key}`)
    .join(" · ");

  return (
    <section className="detail" aria-label="Selected cluster">
      <h3>{formatCoord(cluster.latitude, cluster.longitude)}</h3>
      <dl className="facts">
        <div>
          <dt>Detections</dt>
          <dd>{cluster.hotspot_count}</dd>
        </div>
        <div>
          <dt>Max FRP</dt>
          <dd>{formatFrp(cluster.max_frp)}</dd>
        </div>
        <div>
          <dt>Mean FRP</dt>
          <dd>{formatFrp(cluster.mean_frp)}</dd>
        </div>
        <div>
          <dt>Latest</dt>
          <dd>{formatUtc(cluster.latest_acquisition)}</dd>
        </div>
      </dl>
      <p className="fine">
        {cluster.satellites.map(satelliteLabel).join(", ") || "VIIRS"} · day {day} / night {night}
        <br />
        Confidence {confidence}
      </p>
      <p className="fine">
        FRP is fire radiative power, a satellite measure of instantaneous intensity — not a mapped
        perimeter.
      </p>

      <h4>Fire weather</h4>
      {weatherNote ? <p className="fine">{weatherNote}</p> : null}
      {weather == null && !weatherNote ? <p className="fine">Loading CWFIS stations…</p> : null}
      {weather && !weather.available ? (
        <p className="fine">{weather.detail ?? "CWFIS fire weather is unavailable."}</p>
      ) : null}
      {weather?.available && weather.station_count === 0 ? (
        <p className="fine">{weather.detail ?? "No recent BC station observations."}</p>
      ) : null}
      {nearest ? (
        <div className="station-card">
          <p className="station-name">
            <i style={{ background: fwiColor(nearest.station.fwi) }} />
            {nearest.station.name}
            <span>{formatKm(nearest.distanceKm)} away</span>
          </p>
          <p className="fine">
            FWI {formatNumber(nearest.station.fwi, 1)} ({fwiBand(nearest.station.fwi)}) · ISI{" "}
            {formatNumber(nearest.station.isi, 1)} · BUI {formatNumber(nearest.station.bui, 1)} ·
            FFMC {formatNumber(nearest.station.ffmc, 1)}
          </p>
          <p className="fine">
            Wind {formatNumber(nearest.station.wind_speed_kmh, 1)} km/h at{" "}
            {formatNumber(nearest.station.wind_direction_deg, 0)}° · observed{" "}
            {formatUtc(nearest.station.observed_at)}
          </p>
        </div>
      ) : null}
      <StationToggle
        showStations={showStations}
        onToggleStations={onToggleStations}
        disabled={!weather?.stations.length}
      />

      <h4>Situation report</h4>
      <p className="fine">
        Grok writes from this cluster and the nearest CWFIS station. A trained spread model is not
        connected, so the briefing will not invent a 24-hour forecast.
      </p>
      {grokKnown && !grokReady ? (
        <div className="notice notice-warn" role="status">
          <strong>XAI_API_KEY is not set on the API.</strong>
          <p>The map still works. Add the key to generate a situation report.</p>
        </div>
      ) : null}
      <button type="button" className="report-button" disabled={reportDisabled} onClick={onGenerate}>
        {activeReport?.loading ? "Generating situation report…" : "Generate situation report"}
      </button>
      {activeReport?.error ? (
        <div className="notice notice-error" role="alert">
          <p>{activeReport.error}</p>
        </div>
      ) : null}
      {activeReport?.text ? (
        <div className="report" aria-live="polite">
          <p>{activeReport.text}</p>
          {activeReport.result ? (
            <p className="fine">
              Model {activeReport.result.model}. Approximate footprint{" "}
              {formatNumber(activeReport.result.approximate_area_ha, 1)} ha. Spread model not
              connected.
              {activeReport.result.inputs.station_name
                ? ` Weather from ${activeReport.result.inputs.station_name}, FWI ${formatNumber(activeReport.result.inputs.fwi, 1)}, wind ${formatNumber(activeReport.result.inputs.wind_speed_kmh, 0)} km/h ${activeReport.result.inputs.wind_direction}.`
                : " Fire weather was not attached."}
            </p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
