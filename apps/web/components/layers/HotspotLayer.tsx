"use client";

import { CircleMarker, Tooltip } from "react-leaflet";
import type { HotspotCluster } from "@/lib/types";
import { formatCoord, formatFrp, frpColor, satelliteLabel } from "@/lib/format";

function markerRadius(count: number): number {
  return Math.min(28, 7 + Math.sqrt(count) * 3);
}

type HotspotLayerProps = {
  clusters: HotspotCluster[];
  selectedId: string | null;
  onSelect: (id: string) => void;
};

export default function HotspotLayer({ clusters, selectedId, onSelect }: HotspotLayerProps) {
  return (
    <>
      {clusters.map((cluster) => {
        const active = cluster.id === selectedId;
        const color = frpColor(cluster.max_frp);
        return (
          <CircleMarker
            key={cluster.id}
            center={[cluster.latitude, cluster.longitude]}
            radius={markerRadius(cluster.hotspot_count)}
            eventHandlers={{ click: () => onSelect(cluster.id) }}
            pathOptions={{
              color: active ? "#f6f1e7" : color,
              weight: active ? 3 : 1,
              fillColor: color,
              fillOpacity: 0.88,
            }}
          >
            <Tooltip>
              {formatCoord(cluster.latitude, cluster.longitude)}
              <br />
              {cluster.hotspot_count} detections · max FRP {formatFrp(cluster.max_frp)}
              <br />
              {cluster.satellites.map(satelliteLabel).join(", ") || "VIIRS"}
            </Tooltip>
          </CircleMarker>
        );
      })}
    </>
  );
}
