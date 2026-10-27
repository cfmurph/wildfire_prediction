import { describe, expect, it } from "vitest";
import {
  formatCoord,
  formatFrp,
  formatKm,
  formatNumber,
  formatUtc,
  frpColor,
  fwiBand,
  fwiColor,
  haversineKm,
  nearestStation,
  satelliteLabel,
} from "./format";
import type { WeatherStation } from "./types";

const station = (id: string, lat: number, lon: number): WeatherStation => ({
  id,
  name: id,
  latitude: lat,
  longitude: lon,
  observed_at: "2026-09-26T12:00:00Z",
  temp_c: null,
  relative_humidity: null,
  wind_speed_kmh: null,
  wind_direction_deg: null,
  precip_mm: null,
  ffmc: null,
  dmc: null,
  dc: null,
  bui: null,
  isi: null,
  fwi: 10,
});

describe("format", () => {
  it("labels known satellites and leaves unknown codes", () => {
    expect(satelliteLabel("N")).toBe("Suomi-NPP");
    expect(satelliteLabel("N20")).toBe("NOAA-20");
    expect(satelliteLabel("N21")).toBe("NOAA-21");
    expect(satelliteLabel("X")).toBe("X");
  });

  it("formats coordinates, FRP, numbers, and distance", () => {
    expect(formatCoord(49.1, -123.456)).toBe("49.10°N, 123.46°W");
    expect(formatCoord(-1.2, 10)).toBe("1.20°S, 10.00°E");
    expect(formatFrp(null)).toBe("n/a");
    expect(formatFrp(Number.NaN)).toBe("n/a");
    expect(formatFrp(12.34)).toBe("12.3 MW");
    expect(formatNumber(null)).toBe("n/a");
    expect(formatNumber(1.26, 1)).toBe("1.3");
    expect(formatKm(9.94)).toBe("9.9 km");
    expect(formatKm(10)).toBe("10 km");
    expect(formatKm(10.6)).toBe("11 km");
  });

  it("formats UTC timestamps and returns the raw string when it is not a date", () => {
    expect(formatUtc(null)).toBe("Unknown time");
    const formatted = formatUtc("2026-09-26T19:00:00Z");
    expect(formatted).toContain("26");
    expect(formatted).toContain("UTC");
    expect(formatUtc("not-a-date")).toBe("not-a-date");
  });

  it("bands FWI and FRP on the panel scale", () => {
    expect(fwiBand(null)).toBe("unknown");
    expect(fwiColor(null)).toBe("#8d8a82");
    expect(fwiBand(4.9)).toBe("low");
    expect(fwiBand(5)).toBe("moderate");
    expect(fwiBand(10)).toBe("high");
    expect(fwiBand(20)).toBe("very high");
    expect(fwiBand(30)).toBe("extreme");
    expect(fwiColor(30)).toBe("#6a1b9a");
    expect(frpColor(null)).toBe("#e6c07b");
    expect(frpColor(4.9)).toBe("#e6c07b");
    expect(frpColor(5)).toBe("#f08a24");
    expect(frpColor(20)).toBe("#e23d2b");
  });

  it("picks the nearest station and handles an empty list", () => {
    expect(haversineKm(49, -123, 49, -123)).toBe(0);
    expect(nearestStation(49, -123, [])).toBeNull();
    const found = nearestStation(49.1, -123.1, [
      station("far", 60, -139),
      station("near", 49.11, -123.11),
    ]);
    expect(found?.station.id).toBe("near");
    expect(found && found.distanceKm).toBeLessThan(5);
  });
});
