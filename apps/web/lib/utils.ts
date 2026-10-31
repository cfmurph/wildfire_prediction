import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fwiLabel(fwi: number): string {
  if (fwi < 5)  return "Low";
  if (fwi < 12) return "Moderate";
  if (fwi < 20) return "High";
  if (fwi < 30) return "Very High";
  return "Extreme";
}

export function fwiColor(fwi: number): string {
  if (fwi < 5)  return "text-green-400";
  if (fwi < 12) return "text-yellow-400";
  if (fwi < 20) return "text-orange-400";
  if (fwi < 30) return "text-red-400";
  return "text-purple-400";
}

/** Properties of the nearest CWFIS station within maxKm of (lat, lon), or null. */
export function nearestStation(
  stations: GeoJSON.FeatureCollection,
  lat: number,
  lon: number,
  maxKm = 100,
): Record<string, number> | null {
  const toRad = Math.PI / 180;
  let best: Record<string, number> | null = null;
  let bestKm = maxKm;
  for (const f of stations.features) {
    if (f.geometry.type !== "Point") continue;
    const [sLon, sLat] = f.geometry.coordinates;
    // Equirectangular distance is accurate enough at 100 km
    const x = (sLon - lon) * toRad * Math.cos(((sLat + lat) / 2) * toRad);
    const km = 6371 * Math.hypot(x, (sLat - lat) * toRad);
    if (km <= bestKm) [best, bestKm] = [f.properties as Record<string, number>, km];
  }
  return best;
}
