"use client";

import { useEffect } from "react";
import { CircleMarker, MapContainer, TileLayer, Tooltip, useMap } from "react-leaflet";
import L from "leaflet";
import type { HotspotCluster, WeatherStation } from "@/lib/types";
import { formatCoord, formatFrp, frpColor, fwiColor, satelliteLabel } from "@/lib/format";
import "leaflet/dist/leaflet.css";

type FireMapProps = {
  clusters: HotspotCluster[];
  stations: WeatherStation[];
  showStations: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
};

function FitClusters({ clusters }: { clusters: HotspotCluster[] }) {
  const map = useMap();
  const signature = clusters.map((cluster) => cluster.id).join("|");

  useEffect(() => {
    if (clusters.length === 0) {
      map.setView([54.5, -125.5], 5);
      return;
    }
    const bounds = L.latLngBounds(clusters.map((cluster) => [cluster.latitude, cluster.longitude]));
    map.fitBounds(bounds.pad(0.35), { maxZoom: 7 });
    // signature captures cluster identity without refitting on unrelated renders
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, map]);

  return null;
}

function FlyToSelected({
  latitude,
  longitude,
  token,
}: {
  latitude: number | null;
  longitude: number | null;
  token: string;
}) {
  const map = useMap();

  useEffect(() => {
    if (latitude == null || longitude == null || !token) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    map.flyTo([latitude, longitude], Math.max(map.getZoom(), 7), { duration: reduce ? 0 : 0.7 });
  }, [latitude, longitude, token, map]);

  return null;
}

function markerRadius(count: number): number {
  return Math.min(28, 7 + Math.sqrt(count) * 3);
}

export default function FireMap({
  clusters,
  stations,
  showStations,
  selectedId,
  onSelect,
}: FireMapProps) {
  const selected = clusters.find((cluster) => cluster.id === selectedId) ?? null;

  return (
    <MapContainer
      className="map"
      center={[54.5, -125.5]}
      zoom={5}
      minZoom={4}
      maxZoom={12}
      maxBounds={[
        [46.8, -145],
        [62.2, -110],
      ]}
      maxBoundsViscosity={0.7}
      scrollWheelZoom
    >
      <TileLayer
        attribution='Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community'
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
      />
      <FitClusters clusters={clusters} />
      <FlyToSelected
        token={selected?.id ?? ""}
        latitude={selected?.latitude ?? null}
        longitude={selected?.longitude ?? null}
      />
      {showStations
        ? stations.map((station) => (
            <CircleMarker
              key={`wx-${station.id}`}
              center={[station.latitude, station.longitude]}
              radius={4}
              pathOptions={{
                color: fwiColor(station.fwi),
                weight: 1,
                fillColor: fwiColor(station.fwi),
                fillOpacity: 0.75,
              }}
            >
              <Tooltip>
                {station.name} · FWI {station.fwi ?? "n/a"}
              </Tooltip>
            </CircleMarker>
          ))
        : null}
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
    </MapContainer>
  );
}
