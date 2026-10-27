import { describe, expect, it } from "vitest";
import { activeLayers, DEFAULT_VIEW, MAP_VIEWS, viewById, type ViewId } from "./views";

describe("map views", () => {
  it("marks only Current as available and drops unimplemented layers", () => {
    expect(DEFAULT_VIEW).toBe("current");
    expect(MAP_VIEWS.map((view) => view.id)).toEqual(["history", "risk", "current", "predict"]);
    expect(MAP_VIEWS.filter((view) => view.available).map((view) => view.id)).toEqual(["current"]);
    expect(activeLayers(viewById("current"))).toEqual(["hotspots", "fire-weather"]);
    expect(activeLayers(viewById("history"))).toEqual([]);
    expect(activeLayers(viewById("risk"))).toEqual([]);
    expect(activeLayers(viewById("predict"))).toEqual([]);
    expect(viewById("history").summary).toContain("2012");
  });

  it("rejects an unknown id", () => {
    expect(() => viewById("nope" as ViewId)).toThrow(/Unknown map view/);
  });
});
