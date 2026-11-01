"use client";

import { useEffect, useState } from "react";
import { Flame, Wind, ArrowLeft, ExternalLink } from "lucide-react";
import { SelectedFire } from "@/app/page";
import { cn, fwiColor, fwiLabel, kmBetween, statusColor } from "@/lib/utils";

const API = process.env.NEXT_PUBLIC_API_URL;

interface Props {
  fire: SelectedFire | null;
  hotspots?: GeoJSON.Feature[];
  onBack?: () => void;
}

type Station = { station_name: string; distance_km: number; FWI: number; ISI: number; wind_speed: number; wind_dir: number };
// 24 h areas (ha) for a new fire under the station's weather, from the heuristic spread model
type Spread = { p25: number; p50: number; p75: number; dir: number };

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

export default function FirePanel({ fire, hotspots = [], onBack }: Props) {
  // undefined while loading, null when no station is reporting nearby
  const [station, setStation] = useState<Station | null | undefined>(undefined);
  const [spread, setSpread] = useState<Spread | null>(null);

  useEffect(() => {
    setStation(undefined);
    setSpread(null);
    if (!fire || fire.source !== "active") return;
    let cancelled = false;
    fetch(`${API}/api/v1/weather/nearest?lat=${fire.lat}&lon=${fire.lon}`)
      .then((r) => r.json())
      .then(async (s) => {
        if (cancelled) return;
        if (typeof s?.FWI !== "number") return setStation(null);
        setStation(s);
        const q = `lat=${fire.lat}&lon=${fire.lon}&fwi=${s.FWI}&isi=${s.ISI}&wind_speed=${s.wind_speed / 3.6}&wind_dir=${s.wind_dir}&hours=24`;
        const features: GeoJSON.Feature[] = (await (await fetch(`${API}/api/v1/predict/spread?${q}`)).json())?.features ?? [];
        const area = (p: string) => Number(features.find((f) => f.properties?.percentile === p)?.properties?.area_ha);
        if (!cancelled && area("p50") > 0) {
          setSpread({ p25: area("p25"), p50: area("p50"), p75: area("p75"), dir: Number(features[0].properties?.spread_dir_deg) });
        }
      })
      .catch(() => { if (!cancelled) setStation((s) => s ?? null); });
    return () => { cancelled = true; };
  }, [fire]);

  if (!fire) {
    return (
      <div className="p-4 text-center text-gray-500 text-sm mt-8">
        <Flame className="w-8 h-8 mx-auto mb-2 opacity-30" />
        <p>Click a fire on the map</p>
        <p className="text-xs mt-1 opacity-60">to see its size, cause and dates</p>
      </div>
    );
  }

  const p = fire.properties;
  const status = String(p.fire_status || "");
  // Detections within the fire's rough radius plus 5 km, from its reported area
  const radiusKm = Math.sqrt(fire.size_ha / 100 / Math.PI) + 5;
  const detections = hotspots.filter((h) => {
    const [lon, lat] = (h.geometry as GeoJSON.Point).coordinates;
    return kmBetween(fire.lat, fire.lon, lat, lon) <= radiusKm;
  }).length;

  return (
    <div className="p-4 space-y-4">
      {onBack && (
        <button onClick={onBack} className="flex items-center gap-1 text-xs text-gray-400 hover:text-white transition-colors">
          <ArrowLeft className="w-3.5 h-3.5" /> All fires
        </button>
      )}

      {/* Fire header */}
      <div>
        <h2 className="font-semibold text-sm text-white leading-tight">
          {fire.fire_name || fire.fire_id}
          {fire.fire_name && fire.fire_name !== fire.fire_id && <span className="text-gray-500 font-normal"> · {fire.fire_id}</span>}
          {fire.source === "history" && p.fire_year ? <span className="text-gray-500 font-normal"> · {String(p.fire_year)} season</span> : null}
        </h2>
        <p className="text-xs text-gray-400 mt-0.5">
          {String(p.location || `${fire.lat.toFixed(4)}°N, ${Math.abs(fire.lon).toFixed(4)}°W`)}
        </p>
        {status && (
          <span className="inline-flex items-center gap-1.5 mt-2 text-xs font-medium" style={{ color: statusColor(status) }}>
            <span className="w-2 h-2 rounded-full" style={{ background: statusColor(status) }} />
            {status}{p.fire_of_note ? " · Fire of note" : ""}
          </span>
        )}
      </div>

      {/* Key facts */}
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Size" value={fire.size_ha > 0 ? `${fire.size_ha.toLocaleString()} ha` : "—"} />
        <Stat label="Cause" value={String(p.fire_cause || "—")} />
        <Stat label="Started" value={String(p.ignition_date || "—").slice(0, 10)} />
        {fire.source === "active" && (
          <Stat label="Hotspots, 24 h" value={`${detections} within ${radiusKm.toFixed(0)} km`} />
        )}
      </div>

      {/* Weather from the nearest reporting station */}
      {fire.source === "active" && (
        <div className="space-y-2">
          <p className="text-[10px] uppercase tracking-wide text-gray-500">
            {station ? `Weather · ${station.station_name}, ${station.distance_km} km away` : "Weather"}
          </p>
          {station === undefined ? (
            <p className="text-xs text-gray-500">Loading…</p>
          ) : station === null ? (
            <p className="text-xs text-gray-500">No CWFIS station is reporting within 100 km.</p>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              <Stat label="FWI" value={station.FWI.toFixed(0)} color={fwiColor(station.FWI)} badge={fwiLabel(station.FWI)} />
              <Stat
                label="Wind"
                value={`${station.wind_speed.toFixed(0)} km/h from ${COMPASS[Math.round(station.wind_dir / 45) % 8]}`}
                icon={<Wind className="w-3 h-3" />}
              />
            </div>
          )}
          {spread && (
            <p className="text-xs text-gray-400 leading-relaxed">
              In this weather a new fire would reach about <span className="text-white">{spread.p50.toLocaleString()} ha</span> in
              24 h (likely {spread.p25.toLocaleString()}–{spread.p75.toLocaleString()} ha), spreading {COMPASS[Math.round(spread.dir / 45) % 8]}.
              <span className="block text-[10px] text-gray-500 mt-0.5">
                Rule of thumb from FWI and wind only; it ignores this fire&apos;s size, fuels and terrain.
              </span>
            </p>
          )}
        </div>
      )}

      {fire.source === "active" && typeof p.fire_url === "string" && p.fire_url.startsWith("https://") && (
        <a href={p.fire_url} target="_blank" rel="noopener noreferrer"
           className="flex items-center gap-1 text-xs text-orange-400 hover:text-orange-300">
          Official BC Wildfire Service incident page <ExternalLink className="w-3 h-3" />
        </a>
      )}

      {/* Raw properties (collapsed) */}
      <details className="text-xs text-gray-500">
        <summary className="cursor-pointer hover:text-gray-300 transition-colors">
          Raw properties
        </summary>
        <pre className="mt-2 bg-gray-800 rounded p-2 overflow-x-auto text-[10px] leading-relaxed">
          {JSON.stringify(fire.properties, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function Stat({
  label, value, color, badge, icon,
}: {
  label: string;
  value: string;
  color?: string;
  badge?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="bg-gray-800 rounded-lg p-2.5">
      <p className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</p>
      <p className={cn("text-sm font-semibold mt-0.5 flex items-center gap-1", color ?? "text-white")}>
        {icon}{value}
      </p>
      {badge && (
        <span className={cn("text-[10px] px-1.5 py-0.5 rounded mt-1 inline-block", color ?? "text-gray-400")}>
          {badge}
        </span>
      )}
    </div>
  );
}
