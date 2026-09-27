import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import type { HotspotCluster, WeatherStation } from "@/lib/types";
import FireMap from "./FireMap";
import HotspotLayer from "./layers/HotspotLayer";
import WeatherLayer from "./layers/WeatherLayer";

vi.mock("leaflet/dist/leaflet.css", () => ({}));

vi.mock("leaflet", () => ({
  default: {
    latLngBounds: () => ({
      pad: () => ({}),
    }),
  },
}));

vi.mock("react-leaflet", () => ({
  MapContainer: ({ children }: { children?: ReactNode }) => <div data-testid="map">{children}</div>,
  TileLayer: () => <div data-testid="tiles" />,
  useMap: () => ({
    setView: vi.fn(),
    fitBounds: vi.fn(),
    flyTo: vi.fn(),
    getZoom: () => 5,
  }),
  CircleMarker: ({
    children,
    eventHandlers,
  }: {
    children?: ReactNode;
    eventHandlers?: { click?: () => void };
  }) => (
    <button type="button" onClick={() => eventHandlers?.click?.()}>
      {children}
    </button>
  ),
  Tooltip: ({ children }: { children?: ReactNode }) => <span>{children}</span>,
}));

const cluster: HotspotCluster = {
  id: "49.00,-123.20",
  latitude: 49.1,
  longitude: -123.1,
  hotspot_count: 3,
  max_frp: 22,
  mean_frp: 10,
  max_brightness: 320,
  latest_acquisition: "2026-09-26T19:00:00Z",
  confidence: { high: 3, nominal: 0, low: 0, unknown: 0 },
  satellites: ["N21"],
  daynight: { D: 3, N: 0 },
  bbox: [-123.2, 49.0, -123.1, 49.2],
};

const station: WeatherStation = {
  id: "stn",
  name: "TEST STATION",
  latitude: 49.2,
  longitude: -123.2,
  observed_at: "2026-09-26T12:00:00Z",
  temp_c: 18,
  relative_humidity: 30,
  wind_speed_kmh: 12,
  wind_direction_deg: 180,
  precip_mm: 0,
  ffmc: 90,
  dmc: 20,
  dc: 100,
  bui: 30,
  isi: 6,
  fwi: 21,
};

describe("Leaflet map", () => {
  it("renders hotspots and weather and selects a cluster", () => {
    const onSelect = vi.fn();
    render(
      <FireMap
        activeLayers={["hotspots", "fire-weather"]}
        clusters={[cluster]}
        stations={[station]}
        showStations
        selectedId={null}
        onSelect={onSelect}
      />,
    );
    expect(screen.getByTestId("map")).toBeInTheDocument();
    expect(screen.getByText(/NOAA-21/)).toBeInTheDocument();
    expect(screen.getByText(/TEST STATION/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /NOAA-21/ }));
    expect(onSelect).toHaveBeenCalledWith(cluster.id);
  });

  it("omits hotspot markers when that layer is not active", () => {
    render(
      <FireMap
        activeLayers={["fire-weather"]}
        clusters={[cluster]}
        stations={[station]}
        showStations={false}
        selectedId={null}
        onSelect={vi.fn()}
      />,
    );
    expect(screen.queryByText(/NOAA-21/)).not.toBeInTheDocument();
  });

  it("renders the layer components on their own", () => {
    render(
      <>
        <HotspotLayer clusters={[cluster]} selectedId={cluster.id} onSelect={vi.fn()} />
        <WeatherLayer stations={[station]} />
      </>,
    );
    expect(screen.getAllByRole("button").length).toBeGreaterThan(0);
    expect(screen.getByText(/TEST STATION/)).toBeInTheDocument();
  });
});
