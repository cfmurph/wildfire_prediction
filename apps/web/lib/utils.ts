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

/** BC Wildfire Service stages of control, most urgent first, with map colours. */
export const FIRE_STATUSES = [
  { status: "Out of Control", color: "#ef4444" },
  { status: "Being Held",     color: "#f59e0b" },
  { status: "Under Control",  color: "#22c55e" },
] as const;

export function statusColor(status: string): string {
  return FIRE_STATUSES.find((s) => s.status === status)?.color ?? "#9ca3af";
}

/** Great-circle distance in km (equirectangular; fine at the scale of one fire). */
export function kmBetween(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const rad = Math.PI / 180;
  const x = (lon2 - lon1) * rad * Math.cos(((lat1 + lat2) / 2) * rad);
  return 6371 * Math.hypot(x, (lat2 - lat1) * rad);
}
