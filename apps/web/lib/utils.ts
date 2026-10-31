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
