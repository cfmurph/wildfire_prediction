import type { HealthResponse, HotspotCluster, HotspotsResponse, ReportResponse, WeatherResponse } from "./types";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function apiBase(): string {
  const raw = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (!raw) return "http://localhost:8000";
  return raw.replace(/\/$/, "");
}

function detailOf(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object" || !("detail" in body)) return fallback;
  const detail = (body as { detail: unknown }).detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail == null) return fallback;
  return JSON.stringify(detail);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase()}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  const text = await response.text();
  let body: unknown = {};
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text.slice(0, 300) };
    }
  }
  if (!response.ok) {
    throw new ApiError(response.status, detailOf(body, `Request failed (${response.status})`));
  }
  return body as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getHotspots(): Promise<HotspotsResponse> {
  return request<HotspotsResponse>("/hotspots");
}

export function getWeather(): Promise<WeatherResponse> {
  return request<WeatherResponse>("/weather");
}

export function postReport(cluster: HotspotCluster): Promise<ReportResponse> {
  return request<ReportResponse>("/report", {
    method: "POST",
    body: JSON.stringify({ cluster }),
  });
}
