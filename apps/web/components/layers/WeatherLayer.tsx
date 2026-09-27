"use client";

import { CircleMarker, Tooltip } from "react-leaflet";
import type { WeatherStation } from "@/lib/types";
import { fwiColor } from "@/lib/format";

type WeatherLayerProps = {
  stations: WeatherStation[];
};

export default function WeatherLayer({ stations }: WeatherLayerProps) {
  return (
    <>
      {stations.map((station) => (
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
      ))}
    </>
  );
}
