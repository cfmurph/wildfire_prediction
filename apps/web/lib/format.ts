import type { WeatherStation } from "./types";

const SATELLITES: Record<string, string> = {
  N: "Suomi-NPP",
  N20: "NOAA-20",
  N21: "NOAA-21",
};

export function satelliteLabel(code: string): string {
  return SATELLITES[code] ?? code;
}

export function formatCoord(lat: number, lon: number): string {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(2)}°${ns}, ${Math.abs(lon).toFixed(2)}°${ew}`;
}

export function formatFrp(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "n/a";
  return `${value.toFixed(1)} MW`;
}

export function formatUtc(iso: string | null): string {
  if (!iso) return "Unknown time";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return new Intl.DateTimeFormat("en-CA", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short",
  }).format(date);
}

export function formatNumber(value: number | null, digits = 0): string {
  if (value == null || Number.isNaN(value)) return "n/a";
  return value.toFixed(digits);
}

export function fwiColor(fwi: number | null): string {
  if (fwi == null) return "#8d8a82";
  if (fwi >= 30) return "#6a1b9a";
  if (fwi >= 20) return "#c62828";
  if (fwi >= 10) return "#ef6c00";
  if (fwi >= 5) return "#f9a825";
  return "#2e7d32";
}

export function fwiBand(fwi: number | null): string {
  if (fwi == null) return "unknown";
  if (fwi >= 30) return "extreme";
  if (fwi >= 20) return "very high";
  if (fwi >= 10) return "high";
  if (fwi >= 5) return "moderate";
  return "low";
}

export function frpColor(frp: number | null): string {
  if (frp == null) return "#e6c07b";
  if (frp >= 20) return "#e23d2b";
  if (frp >= 5) return "#f08a24";
  return "#e6c07b";
}

export function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const radius = 6371;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dPhi = toRad(lat2 - lat1);
  const dLambda = toRad(lon2 - lon1);
  const phi1 = toRad(lat1);
  const phi2 = toRad(lat2);
  const a =
    Math.sin(dPhi / 2) ** 2 +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLambda / 2) ** 2;
  return 2 * radius * Math.asin(Math.sqrt(a));
}

export function nearestStation(
  lat: number,
  lon: number,
  stations: WeatherStation[],
): { station: WeatherStation; distanceKm: number } | null {
  let best: WeatherStation | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const station of stations) {
    const distance = haversineKm(lat, lon, station.latitude, station.longitude);
    if (distance < bestDistance) {
      best = station;
      bestDistance = distance;
    }
  }
  if (!best) return null;
  return { station: best, distanceKm: bestDistance };
}

export function formatKm(distanceKm: number): string {
  if (distanceKm < 10) return `${distanceKm.toFixed(1)} km`;
  return `${Math.round(distanceKm)} km`;
}
