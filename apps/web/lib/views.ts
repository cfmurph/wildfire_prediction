/**
 * One map, four views. Only `current` is implemented.
 * Add a layer id here, then render it from `components/layers` when that view is built.
 */

export type ViewId = "history" | "risk" | "current" | "predict";

export type LayerId = "hotspots" | "fire-weather" | "fire-history" | "burn-likelihood" | "spread";

export type MapView = {
  id: ViewId;
  label: string;
  summary: string;
  available: boolean;
  /** Layers drawn while this view is selected. Unimplemented ids are ignored by the map. */
  layers: readonly LayerId[];
};

export const MAP_VIEWS: readonly MapView[] = [
  {
    id: "history",
    label: "History",
    summary: "Past BC fires from 2012–2023, explored with a time slider.",
    available: false,
    layers: ["fire-history"],
  },
  {
    id: "risk",
    label: "Long-term risk",
    summary: "Where British Columbia is more likely to burn over the long term.",
    available: false,
    layers: ["burn-likelihood"],
  },
  {
    id: "current",
    label: "Current",
    summary: "Near-real-time VIIRS hotspots and CWFIS fire weather.",
    available: true,
    layers: ["hotspots", "fire-weather"],
  },
  {
    id: "predict",
    label: "Short-term",
    summary: "Next-day spread for each active fire.",
    available: false,
    layers: ["spread"],
  },
];

export const DEFAULT_VIEW: ViewId = "current";

const IMPLEMENTED_LAYERS = new Set<LayerId>(["hotspots", "fire-weather"]);

export function viewById(id: ViewId): MapView {
  const view = MAP_VIEWS.find((item) => item.id === id);
  if (!view) {
    throw new Error(`Unknown map view: ${id}`);
  }
  return view;
}

export function activeLayers(view: MapView): LayerId[] {
  return view.layers.filter((layer) => IMPLEMENTED_LAYERS.has(layer));
}
