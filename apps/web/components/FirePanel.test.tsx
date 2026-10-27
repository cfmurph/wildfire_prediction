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

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("FirePanel", () => {
  it("asks for a map selection when nothing is selected", () => {
    render(<FirePanel fire={null} view="current" />);
    expect(screen.getByText("Click a fire on the map")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Generate Situation Report/ })).not.toBeInTheDocument();
  });

  it("posts a situation request and shows the briefing", async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      void init;
      return new Response(JSON.stringify({ report: "Grounded briefing." }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<FirePanel fire={fire} view="current" />);
    const button = screen.getByRole("button", { name: /Generate Situation Report/ });
    expect(button).toBeEnabled();
    fireEvent.click(button);
    await waitFor(() => expect(screen.getByText("Grounded briefing.")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/situation",
      expect.objectContaining({ method: "POST" }),
    );
    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body));
    expect(body.fire_id).toBe("N123");
    expect(body.lat).toBe(49.25);
    expect(body.fwi).toBe(18);
  });

  it("disables the button while the request is in flight and shows API errors", async () => {
    let resolveFetch: (response: Response) => void = () => undefined;
    const fetchMock = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<FirePanel fire={fire} view="predictions" />);
    const button = screen.getByRole("button", { name: /Generate Situation Report/ });
    fireEvent.click(button);
    await waitFor(() => expect(button).toBeDisabled());
    resolveFetch(new Response("nope", { status: 500 }));
    await waitFor(() => expect(screen.getByText("API 500")).toBeInTheDocument());
    expect(button).toBeEnabled();
  });
});
