import { describe, expect, it } from "vitest";
import { fwiColor, fwiLabel, kmBetween, statusColor } from "./utils";

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

describe("statusColor", () => {
  it("maps BCWS stages of control and greys out anything else", () => {
    expect(statusColor("Out of Control")).toBe("#ef4444");
    expect(statusColor("Under Control")).toBe("#22c55e");
    expect(statusColor("Unknown")).toBe("#9ca3af");
  });
});

describe("kmBetween", () => {
  it("measures about 111 km per degree of latitude", () => {
    expect(kmBetween(50, -120, 51, -120)).toBeCloseTo(111.2, 0);
    expect(kmBetween(50, -120, 50, -120)).toBe(0);
  });
});
