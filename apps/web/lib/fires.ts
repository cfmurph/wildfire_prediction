import type { SelectedFire } from "@/app/page";

export type CurrentData = {
  fires: GeoJSON.Feature[] | null;  // null when the fire data could not be loaded
  hotspots: GeoJSON.Feature[];
  stations: number;
  updated: Date;
};

/** SelectedFire from a BCWS active-fire feature. */
export function activeFire(f: GeoJSON.Feature): SelectedFire {
  const props = (f.properties ?? {}) as Record<string, unknown>;
  const [lon, lat] = (f.geometry as GeoJSON.Point).coordinates;
  return {
    fire_id: String(props.fire_number || "unknown"),
    fire_name: String(props.fire_name || ""),
    lat, lon,
    size_ha: Number(props.size_ha) || 0,
    source: "active",
    properties: props,
  };
}
