import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiBase, getHealth, getHotspots, getWeather, postReport } from "./api";
import type { HotspotCluster } from "./types";

const cluster = { id: "49.00,-123.20" } as HotspotCluster;

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("apiBase", () => {
  it("defaults and strips a trailing slash", () => {
    vi.stubEnv("NEXT_PUBLIC_API_URL", "");
    expect(apiBase()).toBe("http://localhost:8000");
    vi.stubEnv("NEXT_PUBLIC_API_URL", "   ");
    expect(apiBase()).toBe("http://localhost:8000");
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example/");
    expect(apiBase()).toBe("https://api.example");
  });
});

describe("request", () => {
  it("loads health, hotspots, and weather", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url).endsWith("/health")) {
        return new Response(JSON.stringify({ status: "ok", region: "BC" }), { status: 200 });
      }
      if (String(url).endsWith("/hotspots")) {
        return new Response(JSON.stringify({ clusters: [] }), { status: 200 });
      }
      return new Response(JSON.stringify({ stations: [], available: true }), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example");
    await expect(getHealth()).resolves.toMatchObject({ region: "BC" });
    await expect(getHotspots()).resolves.toMatchObject({ clusters: [] });
    await expect(getWeather()).resolves.toMatchObject({ available: true });
    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      "https://api.example/health",
      "https://api.example/hotspots",
      "https://api.example/weather",
    ]);
  });

  it("posts the cluster and surfaces string, structured, and non-JSON errors", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ report: "ok" }), { status: 200 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "FIRMS_MAP_KEY is not configured" }), { status: 503 }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: [{ msg: "field required" }] }), { status: 422 }),
      )
      .mockResolvedValueOnce(new Response("upstream exploded", { status: 502 }))
      .mockResolvedValueOnce(new Response("", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    vi.stubEnv("NEXT_PUBLIC_API_URL", "https://api.example");

    await expect(postReport(cluster)).resolves.toMatchObject({ report: "ok" });
    const [, init] = fetchMock.mock.calls[0];
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ cluster });
    expect((init?.headers as Record<string, string>)["Content-Type"]).toBe("application/json");

    await expect(postReport(cluster)).rejects.toMatchObject({
      name: "ApiError",
      status: 503,
      message: "FIRMS_MAP_KEY is not configured",
    });
    await expect(getHealth()).rejects.toThrow(/field required/);
    const textError = await getHotspots().catch((error: unknown) => error);
    expect(textError).toBeInstanceOf(ApiError);
    expect((textError as ApiError).status).toBe(502);
    expect((textError as ApiError).message).toContain("upstream exploded");
    await expect(getWeather()).resolves.toEqual({});
  });
});
