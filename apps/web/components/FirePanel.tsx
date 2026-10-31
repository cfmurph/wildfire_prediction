"use client";

import { useEffect, useState } from "react";
import { Flame, Wind, Thermometer, Loader2, AlertTriangle, ArrowLeft, ExternalLink } from "lucide-react";
import { View } from "./ViewSwitcher";
import { SelectedFire } from "@/app/page";
import { cn, fwiColor, fwiLabel, kmBetween, statusColor } from "@/lib/utils";

const API = process.env.NEXT_PUBLIC_API_URL;

interface Props {
  fire: SelectedFire | null;
  view: View;
  hotspots?: GeoJSON.Feature[];
  onBack?: () => void;
}

type Station = { station_name: string; distance_km: number; FWI: number; wind_speed: number; wind_dir: number };

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

export default function FirePanel({ fire, view, hotspots = [], onBack }: Props) {
  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // undefined while loading, null when no station is reporting nearby
  const [station, setStation] = useState<Station | null | undefined>(undefined);
  const [reportsAvailable, setReportsAvailable] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/v1/situation/available`)
      .then((r) => r.json())
      .then((d) => setReportsAvailable(d?.available === true))
      .catch(() => setReportsAvailable(false));
  }, []);

  useEffect(() => {
    setReport(null);
    setError(null);
    setStation(undefined);
    if (!fire || fire.source !== "active") return;
    let cancelled = false;
    fetch(`${API}/api/v1/weather/nearest?lat=${fire.lat}&lon=${fire.lon}`)
      .then((r) => r.json())
      .then((s) => { if (!cancelled) setStation(typeof s?.FWI === "number" ? s : null); })
      .catch(() => { if (!cancelled) setStation(null); });
    return () => { cancelled = true; };
  }, [fire]);

  async function generateReport() {
    if (!fire) return;
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const res = await fetch(`${API}/api/v1/situation`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fire_id: fire.fire_id,
          fire_name: fire.fire_name,
          lat: fire.lat,
          lon: fire.lon,
          current_area_ha: fire.size_ha,
        }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      setReport(data.report);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  if (!fire) {
    return (
      <div className="p-4 text-center text-gray-500 text-sm mt-8">
        <Flame className="w-8 h-8 mx-auto mb-2 opacity-30" />
        <p>Click a fire on the map</p>
        <p className="text-xs mt-1 opacity-60">to see details and generate a situation report</p>
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
        {fire.source === "active" && <Stat label="Started" value={String(p.ignition_date || "—")} />}
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
        </div>
      )}

      {fire.source === "active" && typeof p.fire_url === "string" && p.fire_url.startsWith("https://") && (
        <a href={p.fire_url} target="_blank" rel="noopener noreferrer"
           className="flex items-center gap-1 text-xs text-orange-400 hover:text-orange-300">
          Official BC Wildfire Service incident page <ExternalLink className="w-3 h-3" />
        </a>
      )}

      {/* AI situation report, only when the API has a key */}
      {reportsAvailable && (
        <div>
          <button
            onClick={generateReport}
            disabled={loading}
            className={cn(
              "w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-medium transition-colors",
              loading
                ? "bg-gray-700 text-gray-400 cursor-not-allowed"
                : "bg-orange-600 hover:bg-orange-500 text-white"
            )}
          >
            {loading ? (
              <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Generating…</>
            ) : (
              <><Flame className="w-3.5 h-3.5" /> Generate Situation Report</>
            )}
          </button>

          {error && (
            <div className="mt-2 flex gap-2 text-xs text-red-400 bg-red-950/30 rounded p-2">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              {error}
            </div>
          )}

          {report && (
            <div className="mt-3 text-xs text-gray-300 bg-gray-800 rounded-lg p-3 leading-relaxed whitespace-pre-wrap">
              {renderEmphasis(report)}
            </div>
          )}
        </div>
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

/** Render the **bold** and _italic_ the briefings use, without injecting HTML. */
function renderEmphasis(text: string) {
  return text.split(/(\*\*[^*]+\*\*|(?<!\w)_[^\n]+?_(?!\w))/).map((part, i) =>
    part.startsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong>
    : part.startsWith("_") && part.length > 1 ? <em key={i}>{part.slice(1, -1)}</em>
    : part
  );
}
