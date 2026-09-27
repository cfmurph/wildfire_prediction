"use client";

import { useEffect } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import HotspotLayer from "@/components/layers/HotspotLayer";
import WeatherLayer from "@/components/layers/WeatherLayer";
import type { LayerId } from "@/lib/views";
import type { HotspotCluster, WeatherStation } from "@/lib/types";
import "leaflet/dist/leaflet.css";

type FireMapProps = {
  activeLayers: readonly LayerId[];
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
    const bounds = L.latLngBounds(
      clusters.map((cluster) => [cluster.latitude, cluster.longitude] as [number, number]),
    );
    map.fitBounds(bounds.pad(0.35), { maxZoom: 7 });
  }, [signature, map, clusters]);

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

export default function FireMap({
  activeLayers,
  clusters,
  stations,
  showStations,
  selectedId,
  onSelect,
}: FireMapProps) {
  const showHotspots = activeLayers.includes("hotspots");
  const showWeather = activeLayers.includes("fire-weather") && showStations;
  const visibleClusters = showHotspots ? clusters : [];
  const selected = visibleClusters.find((cluster) => cluster.id === selectedId) ?? null;

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
      <FitClusters clusters={visibleClusters} />
      <FlyToSelected
        token={selected?.id ?? ""}
        latitude={selected?.latitude ?? null}
        longitude={selected?.longitude ?? null}
      />
      {showWeather ? <WeatherLayer stations={stations} /> : null}
      {showHotspots ? (
        <HotspotLayer clusters={visibleClusters} selectedId={selectedId} onSelect={onSelect} />
      ) : null}
    </MapContainer>
  );
}
