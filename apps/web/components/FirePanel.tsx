"use client";

import { useState } from "react";
import { Flame, Wind, Thermometer, Loader2, AlertTriangle } from "lucide-react";
import { View } from "./ViewSwitcher";
import { SelectedFire } from "@/app/page";
import { cn, fwiColor, fwiLabel } from "@/lib/utils";

const API = process.env.NEXT_PUBLIC_API_URL;

interface Props {
  fire: SelectedFire | null;
  view: View;
}

export default function FirePanel({ fire, view }: Props) {
  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
          fwi: fire.fwi ?? 20,
          isi: fire.isi ?? 8,
          wind_speed_ms: fire.wind_speed_ms ?? 5,
          wind_dir_deg: fire.wind_dir_deg ?? 270,
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

  const fwi = fire.fwi ?? 0;

  return (
    <div className="p-4 space-y-4">
      {/* Fire header */}
      <div>
        <h2 className="font-semibold text-sm text-white leading-tight">
          {fire.fire_name || fire.fire_id}
        </h2>
        <p className="text-xs text-gray-400 mt-0.5">
          {fire.lat.toFixed(4)}°N, {Math.abs(fire.lon).toFixed(4)}°W
        </p>
      </div>

      {/* Key stats */}
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Size" value={fire.size_ha > 0 ? `${fire.size_ha.toLocaleString()} ha` : "—"} />
        <Stat
          label="FWI"
          value={fwi > 0 ? fwi.toFixed(0) : "—"}
          color={fwiColor(fwi)}
          badge={fwi > 0 ? fwiLabel(fwi) : undefined}
        />
        <Stat
          label="Wind"
          value={fire.wind_speed_ms ? `${(fire.wind_speed_ms * 3.6).toFixed(0)} km/h` : "—"}
          icon={<Wind className="w-3 h-3" />}
        />
        <Stat
          label="Direction"
          value={fire.wind_dir_deg != null ? `${fire.wind_dir_deg.toFixed(0)}°` : "—"}
        />
      </div>

      {/* Situation report */}
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
            {report}
          </div>
        )}
      </div>

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
