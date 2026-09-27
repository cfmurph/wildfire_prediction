import { describe, expect, it } from "vitest";
import { fwiColor, fwiLabel } from "./utils";

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
