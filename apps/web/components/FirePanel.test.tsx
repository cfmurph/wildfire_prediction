import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SelectedFire } from "@/app/page";
import FirePanel from "./FirePanel";

const fire: SelectedFire = {
  fire_id: "N123",
  fire_name: "Test Fire",
  lat: 49.25,
  lon: -123.1,
  size_ha: 42,
  fwi: 18,
  isi: 6,
  wind_speed_ms: 5,
  wind_dir_deg: 270,
  source: "active",
  properties: { fire_number: "N123" },
};

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

/** fetch stub: AI reports enabled, no nearby station, and `situation` answers report requests. */
function stubApi(situation: (init?: RequestInit) => Promise<Response>, available = true) {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/situation/available")) return json({ available });
    if (url.includes("/weather/nearest")) return json(null);
    return situation(init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("FirePanel", () => {
  it("asks for a map selection when nothing is selected", () => {
    render(<FirePanel fire={null} />);
    expect(screen.getByText("Click a fire on the map")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Generate Situation Report/ })).not.toBeInTheDocument();
  });

  it("posts a situation request and shows the briefing", async () => {
    const fetchMock = stubApi(async () => json({ report: "Grounded briefing." }));
    render(<FirePanel fire={fire} />);
    const button = await screen.findByRole("button", { name: /Generate Situation Report/ });
    expect(button).toBeEnabled();
    fireEvent.click(button);
    await waitFor(() => expect(screen.getByText("Grounded briefing.")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/situation",
      expect.objectContaining({ method: "POST" }),
    );
    const situation = fetchMock.mock.calls.find(([url]) => url.endsWith("/situation"));
    const body = JSON.parse(String(situation?.[1]?.body));
    expect(body.fire_id).toBe("N123");
    expect(body.lat).toBe(49.25);
    // The API looks up station weather itself; the browser must not supply it
    expect(body.fwi).toBeUndefined();
  });

  it("disables the button while the request is in flight and shows API errors", async () => {
    let resolveFetch: (response: Response) => void = () => undefined;
    stubApi(() => new Promise<Response>((resolve) => { resolveFetch = resolve; }));
    render(<FirePanel fire={fire} />);
    const button = await screen.findByRole("button", { name: /Generate Situation Report/ });
    fireEvent.click(button);
    await waitFor(() => expect(button).toBeDisabled());
    resolveFetch(new Response("nope", { status: 500 }));
    await waitFor(() => expect(screen.getByText("API 500")).toBeInTheDocument());
    expect(button).toBeEnabled();
  });
});

describe("FirePanel report formatting", () => {
  it("renders bold and italic without breaking on underscores in names", async () => {
    stubApi(async () => json({ report: "**N123** — 42 ha\n\n_Set XAI_API_KEY for briefings._" }));
    render(<FirePanel fire={fire} />);
    fireEvent.click(await screen.findByRole("button", { name: /Generate Situation Report/ }));
    expect(await screen.findByText("N123", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByText("Set XAI_API_KEY for briefings.", { selector: "em" })).toBeInTheDocument();
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

  it("estimates 24 h spread from the station's weather", async () => {
    const station = { station_name: "LILLOOET", distance_km: 12.3, FWI: 24, ISI: 9, wind_speed: 18, wind_dir: 270 };
    const spread = {
      features: ["p25", "p50", "p75"].map((percentile, i) => ({
        properties: { percentile, area_ha: [15, 62, 200][i], spread_dir_deg: 90 },
      })),
    };
    const fetchMock = vi.fn(async (url: string) => json(url.includes("/predict/spread") ? spread : station));
    vi.stubGlobal("fetch", fetchMock);
    render(<FirePanel fire={fire} />);
    expect(await screen.findByText("62 ha")).toBeInTheDocument();
    expect(screen.getByText(/likely 15–200 ha\), spreading E/)).toBeInTheDocument();
    const url = String(fetchMock.mock.calls.find(([u]) => u.includes("/predict/spread"))?.[0]);
    expect(url).toContain("fwi=24&isi=9&wind_speed=5&wind_dir=270");
  });

  it("says so when no station is reporting", async () => {
    vi.stubGlobal("fetch", respond(null));
    render(<FirePanel fire={fire} />);
    expect(await screen.findByText("No CWFIS station is reporting within 100 km.")).toBeInTheDocument();
  });
});

describe("FirePanel without AI reports", () => {
  it("hides the report button and counts hotspots near the fire", async () => {
    stubApi(async () => json({}), false);
    const point = (lon: number, lat: number): GeoJSON.Feature =>
      ({ type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties: {} });
    // 42 ha fire: radius ~0.4 km + 5 km. One detection 2 km away, one 50 km away.
    render(<FirePanel fire={fire} hotspots={[point(-123.1, 49.268), point(-123.1, 49.7)]} />);
    expect(await screen.findByText("No CWFIS station is reporting within 100 km.")).toBeInTheDocument();
    expect(screen.getByText("1 within 5 km")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Generate Situation Report/ })).not.toBeInTheDocument();
  });
});
