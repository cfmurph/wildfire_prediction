import { describe, expect, it } from "vitest";
import { fwiColor, fwiLabel, nearestStation } from "./utils";

describe("mounted FWI scale", () => {
  it("uses different moderate/high cutovers than lib/format.ts", () => {
    expect(fwiLabel(4.9)).toBe("Low");
    expect(fwiLabel(5)).toBe("Moderate");
    expect(fwiLabel(11.9)).toBe("Moderate");
    expect(fwiLabel(12)).toBe("High");
    expect(fwiLabel(20)).toBe("Very High");
    expect(fwiLabel(30)).toBe("Extreme");
    expect(fwiColor(4)).toBe("text-green-400");
    expect(fwiColor(12)).toBe("text-orange-400");
    expect(fwiColor(30)).toBe("text-purple-400");
  });
});

describe("nearestStation", () => {
  const station = (lon: number, lat: number, FWI: number): GeoJSON.Feature => ({
    type: "Feature",
    geometry: { type: "Point", coordinates: [lon, lat] },
    properties: { FWI },
  });
  const stations: GeoJSON.FeatureCollection = {
    type: "FeatureCollection",
    features: [station(-120.0, 50.0, 10), station(-120.5, 50.0, 30)],
  };

  it("picks the closest station within range", () => {
    expect(nearestStation(stations, 50.0, -120.4)?.FWI).toBe(30);
  });

  it("returns null when no station is within range", () => {
    expect(nearestStation(stations, 55.0, -120.0)).toBeNull();
  });
});
