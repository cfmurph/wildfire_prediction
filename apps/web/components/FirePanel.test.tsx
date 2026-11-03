import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SelectedFire } from "@/app/page";
import FirePanel from "./FirePanel";

const fire: SelectedFire = {
  fire_id: "N123",
  fire_name: "Test Fire",
  lat: 49.25,
  lon: -123.1,
  size_ha: 42,
  source: "active",
  properties: { fire_number: "N123" },
};

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("FirePanel", () => {
  it("asks for a map selection when nothing is selected", () => {
    render(<FirePanel fire={null} />);
    expect(screen.getByText("Click a fire on the map")).toBeInTheDocument();
  });
});

describe("FirePanel station weather", () => {
  const respond = (station: unknown) =>
    vi.fn(async () => new Response(JSON.stringify(station), { status: 200 }));

  it("shows the nearest station's FWI and wind", async () => {
    vi.stubGlobal("fetch", respond({ station_name: "LILLOOET", distance_km: 12.3, FWI: 24, wind_speed: 18, wind_dir: 225 }));
    render(<FirePanel fire={fire} />);
    expect(await screen.findByText("Weather · LILLOOET, 12.3 km away")).toBeInTheDocument();
    expect(screen.getByText("24")).toBeInTheDocument();
    expect(screen.getByText("18 km/h from SW")).toBeInTheDocument();
  });

  it("says so when no station is reporting", async () => {
    vi.stubGlobal("fetch", respond(null));
    render(<FirePanel fire={fire} />);
    expect(await screen.findByText("No CWFIS station is reporting within 100 km.")).toBeInTheDocument();
  });
});

describe("FirePanel hotspots", () => {
  it("counts hotspots near the fire", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json(null)));
    const point = (lon: number, lat: number): GeoJSON.Feature =>
      ({ type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties: {} });
    // 42 ha fire: radius ~0.4 km + 5 km. One detection 2 km away, one 50 km away.
    render(<FirePanel fire={fire} hotspots={[point(-123.1, 49.268), point(-123.1, 49.7)]} />);
    expect(await screen.findByText("No CWFIS station is reporting within 100 km.")).toBeInTheDocument();
    expect(screen.getByText("1 within 5 km")).toBeInTheDocument();
  });
});
