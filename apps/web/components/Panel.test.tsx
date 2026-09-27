import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { HealthResponse, HotspotCluster, HotspotsResponse, WeatherResponse } from "@/lib/types";
import Panel from "./Panel";

const cluster: HotspotCluster = {
  id: "49.00,-123.20",
  latitude: 49.1,
  longitude: -123.1,
  hotspot_count: 2,
  max_frp: 12.5,
  mean_frp: 6,
  max_brightness: 300,
  latest_acquisition: "2026-09-26T19:00:00Z",
  confidence: { high: 1, nominal: 1, low: 0, unknown: 0 },
  satellites: ["N"],
  daynight: { D: 1, N: 1 },
  bbox: [-123.2, 49.0, -123.1, 49.2],
};

const hotspots: HotspotsResponse = {
  region: "BC",
  bbox: { west: -139.06, south: 48.3, east: -114.03, north: 60 },
  sources: ["VIIRS_SNPP_NRT"],
  day_range: 2,
  fetched_at: "2026-09-26T20:00:00Z",
  cached: false,
  hotspot_count: 2,
  cluster_count: 1,
  clusters: [cluster],
  warnings: [],
};

const health = (grok: boolean): HealthResponse => ({
  status: "ok",
  region: "BC",
  firms_configured: true,
  grok_configured: grok,
});

const weather: WeatherResponse = {
  source: "CWFIS",
  available: true,
  detail: null,
  fetched_at: "2026-09-26T20:00:00Z",
  cached: false,
  station_count: 1,
  stations: [
    {
      id: "1",
      name: "NEAR",
      latitude: 49.11,
      longitude: -123.11,
      observed_at: "2026-09-26T12:00:00Z",
      temp_c: 20,
      relative_humidity: 30,
      wind_speed_kmh: 10,
      wind_direction_deg: 180,
      precip_mm: 0,
      ffmc: 90,
      dmc: 20,
      dc: 200,
      bui: 30,
      isi: 5,
      fwi: 12,
    },
  ],
};

function renderPanel(overrides: Partial<Parameters<typeof Panel>[0]> = {}) {
  const onGenerate = vi.fn();
  render(
    <Panel
      hotspotStatus="ready"
      hotspotMessage={null}
      hotspots={hotspots}
      weather={weather}
      weatherNote={null}
      health={health(true)}
      selectedId={cluster.id}
      onSelect={vi.fn()}
      showStations={false}
      onToggleStations={vi.fn()}
      report={null}
      onGenerate={onGenerate}
      {...overrides}
    />,
  );
  return { onGenerate };
}

describe("Panel", () => {
  it("disables the situation report when the API has no XAI key", () => {
    renderPanel({ health: health(false) });
    expect(screen.getByText(/XAI_API_KEY is not set/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Generate situation report/ })).toBeDisabled();
  });

  it("enables the report when Grok is configured and a cluster is selected", () => {
    const { onGenerate } = renderPanel();
    const button = screen.getByRole("button", { name: /Generate situation report/ });
    expect(button).toBeEnabled();
    fireEvent.click(button);
    expect(onGenerate).toHaveBeenCalledOnce();
    expect(screen.getByText("NEAR")).toBeInTheDocument();
  });

  it("hides the report button until a cluster is selected and while health is unknown it stays enabled", () => {
    renderPanel({ selectedId: null, health: null });
    expect(screen.queryByRole("button", { name: /Generate situation report/ })).not.toBeInTheDocument();
    expect(screen.getByText(/Select a cluster/)).toBeInTheDocument();
  });

  it("disables the button while a report is generating", () => {
    renderPanel({
      report: { clusterId: cluster.id, loading: true },
    });
    expect(screen.getByRole("button", { name: /Generating situation report/ })).toBeDisabled();
  });

  it("shows a missing FIRMS key and hotspot errors", () => {
    const { rerender } = render(
      <Panel
        hotspotStatus="missing-key"
        hotspotMessage="Add FIRMS_MAP_KEY."
        hotspots={null}
        weather={null}
        weatherNote={null}
        health={health(false)}
        selectedId={null}
        onSelect={vi.fn()}
        showStations={false}
        onToggleStations={vi.fn()}
        report={null}
        onGenerate={vi.fn()}
      />,
    );
    expect(screen.getByText(/NASA FIRMS key required/)).toBeInTheDocument();
    rerender(
      <Panel
        hotspotStatus="error"
        hotspotMessage="upstream down"
        hotspots={null}
        weather={null}
        weatherNote={null}
        health={health(false)}
        selectedId={null}
        onSelect={vi.fn()}
        showStations={false}
        onToggleStations={vi.fn()}
        report={null}
        onGenerate={vi.fn()}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("upstream down");
  });
});
