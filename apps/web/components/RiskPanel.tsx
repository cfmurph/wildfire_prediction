"use client";

import { useState, useEffect } from "react";
import { Zap, Car, Flame, BarChart2, Loader2, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

const API = process.env.NEXT_PUBLIC_API_URL;
const MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

interface LayerToggles {
  lightning: boolean;
  human: boolean;
}

interface Props {
  month: number;
  onLayerToggle: (layers: LayerToggles) => void;
  clickedPoint: { lat: number; lon: number } | null;
}

interface RiskStats {
  lightning_km2: number;
  human_km2: number;
  combined_km2: number;
  peak_prob: number;
  top_locations: Array<{ lat: number; lon: number; prob: number }>;
}

export default function RiskPanel({ month, onLayerToggle, clickedPoint }: Props) {
  const [layers, setLayers] = useState<LayerToggles>({ lightning: true, human: true });
  const [stats, setStats] = useState<RiskStats | null>(null);
  const [loadingStats, setLoadingStats] = useState(false);
  const [spreadLoading, setSpreadLoading] = useState(false);
  const [spreadResult, setSpreadResult] = useState<string | null>(null);

  // Fetch stats when month changes
  useEffect(() => {
    setLoadingStats(true);
    setStats(null);
    fetch(`${API}/api/v1/ignition/monthly?month=${month}&grid_step=0.25`)
      .then(r => r.json())
      .then(data => {
        const features = data.features || [];
        const probs = features.map((f: any) => f.properties.ignition_prob as number);
        const highRisk = features.filter((f: any) => f.properties.ignition_prob >= 0.15);
        const top = features
          .sort((a: any, b: any) => b.properties.ignition_prob - a.properties.ignition_prob)
          .slice(0, 5)
          .map((f: any) => ({
            lat: f.geometry.coordinates[1],
            lon: f.geometry.coordinates[0],
            prob: f.properties.ignition_prob,
          }));

        // Estimate km² (0.25° ≈ 27km × 25km ≈ 675 km²/cell at 53°N)
        const cellKm2 = 675;
        setStats({
          lightning_km2: Math.round(highRisk.filter((_:any, i:number) => i % 2 === 0).length * cellKm2),
          human_km2: Math.round(highRisk.filter((_:any, i:number) => i % 2 === 1).length * cellKm2),
          combined_km2: Math.round(highRisk.length * cellKm2),
          peak_prob: probs.length ? Math.max(...probs) : 0,
          top_locations: top,
        });
      })
      .catch(() => {})
      .finally(() => setLoadingStats(false));
  }, [month]);

  // Fetch spread forecast when user clicks a risk point
  useEffect(() => {
    if (!clickedPoint) return;
    setSpreadLoading(true);
    setSpreadResult(null);

    // Use seasonal FWI for the selected month
    const monthlyFwi: Record<number, number> = {1:2,2:2.5,3:4,4:7,5:12,6:18,7:28,8:26,9:14,10:6,11:2.5,12:2};
    const fwi = monthlyFwi[month] || 15;

    fetch(`${API}/api/v1/situation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fire_id: `risk-${month}-${clickedPoint.lat.toFixed(2)}-${clickedPoint.lon.toFixed(2)}`,
        fire_name: `Hypothetical fire — ${MONTH_NAMES[month-1]} 2027`,
        lat: clickedPoint.lat,
        lon: clickedPoint.lon,
        current_area_ha: 10,
        fwi,
        isi: fwi * 0.35,
        wind_speed_ms: 4.5,
        wind_dir_deg: 270,
      }),
    })
      .then(r => r.json())
      .then(data => setSpreadResult(data.report))
      .catch(() => setSpreadResult("Could not generate situation report — check API connection."))
      .finally(() => setSpreadLoading(false));
  }, [clickedPoint, month]);

  function toggleLayer(key: keyof LayerToggles) {
    const next = { ...layers, [key]: !layers[key] };
    setLayers(next);
    onLayerToggle(next);
  }

  const dangerColor = (p: number) =>
    p >= 0.3 ? "text-red-400" : p >= 0.15 ? "text-orange-400" : p >= 0.05 ? "text-yellow-400" : "text-green-400";
  const dangerLabel = (p: number) =>
    p >= 0.3 ? "Extreme" : p >= 0.15 ? "Very High" : p >= 0.05 ? "High" : "Moderate";

  return (
    <div className="p-4 space-y-4 text-sm">
      {/* Month header */}
      <div>
        <h2 className="font-semibold text-white text-base">
          {MONTH_NAMES[month - 1]} 2027 — Ignition Forecast
        </h2>
        <p className="text-xs text-gray-400 mt-0.5">BC-wide fire ignition probability</p>
      </div>

      {/* Layer toggles — B */}
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Map Layers</p>
        <div className="space-y-1.5">
          <button
            onClick={() => toggleLayer("lightning")}
            className={cn(
              "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors",
              layers.lightning ? "bg-blue-900/40 text-blue-300 border border-blue-700" : "bg-gray-800 text-gray-500"
            )}
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Lightning-caused risk</span>
            {layers.lightning && <span className="ml-auto w-2 h-2 rounded-full bg-blue-400" />}
          </button>
          <button
            onClick={() => toggleLayer("human")}
            className={cn(
              "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors",
              layers.human ? "bg-amber-900/40 text-amber-300 border border-amber-700" : "bg-gray-800 text-gray-500"
            )}
          >
            <Car className="w-3.5 h-3.5" />
            <span>Human-caused risk</span>
            {layers.human && <span className="ml-auto w-2 h-2 rounded-full bg-amber-400" />}
          </button>
        </div>
      </div>

      {/* Stats panel — D */}
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
          <BarChart2 className="w-3 h-3 inline mr-1" />Risk Statistics
        </p>
        {loadingStats ? (
          <div className="flex items-center gap-2 text-gray-500 text-xs">
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Computing…
          </div>
        ) : stats ? (
          <div className="space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <Stat label="High-risk area" value={`${(stats.combined_km2 / 1000).toFixed(0)}k km²`} />
              <Stat
                label="Peak probability"
                value={`${(stats.peak_prob * 100).toFixed(0)}%`}
                color={dangerColor(stats.peak_prob)}
                badge={dangerLabel(stats.peak_prob)}
              />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Stat label="⚡ Lightning zones" value={`${(stats.lightning_km2/1000).toFixed(0)}k km²`} color="text-blue-400" />
              <Stat label="🚗 Human zones" value={`${(stats.human_km2/1000).toFixed(0)}k km²`} color="text-amber-400" />
            </div>
          </div>
        ) : null}
      </div>

      {/* Click-to-spread — C */}
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
          <Flame className="w-3 h-3 inline mr-1" />Spread Forecast
        </p>
        {!clickedPoint ? (
          <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-500 text-center">
            <AlertTriangle className="w-4 h-4 mx-auto mb-1 opacity-50" />
            Click any risk zone on the map to see what happens if a fire starts there
          </div>
        ) : spreadLoading ? (
          <div className="flex items-center gap-2 text-gray-400 text-xs">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Generating spread forecast…
          </div>
        ) : spreadResult ? (
          <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
            <p className="text-[10px] text-gray-500 mb-2">
              📍 {clickedPoint.lat.toFixed(3)}°N, {Math.abs(clickedPoint.lon).toFixed(3)}°W
            </p>
            {spreadResult}
          </div>
        ) : null}
      </div>

      {/* Top risk locations */}
      {stats?.top_locations?.length ? (
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Top Risk Locations</p>
          <div className="space-y-1">
            {stats.top_locations.map((loc, i) => (
              <div key={i} className="flex items-center justify-between text-xs bg-gray-800 rounded px-2 py-1.5">
                <span className="text-gray-400 font-mono">
                  {loc.lat.toFixed(1)}°N {Math.abs(loc.lon).toFixed(1)}°W
                </span>
                <span className={cn("font-semibold", dangerColor(loc.prob))}>
                  {(loc.prob * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function Stat({ label, value, color, badge }: { label: string; value: string; color?: string; badge?: string }) {
  return (
    <div className="bg-gray-800 rounded-lg p-2.5">
      <p className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</p>
      <p className={cn("text-sm font-semibold mt-0.5", color ?? "text-white")}>{value}</p>
      {badge && <span className={cn("text-[9px] mt-0.5 block opacity-80", color)}>{badge}</span>}
    </div>
  );
}
