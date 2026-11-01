"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import ViewSwitcher, { View } from "@/components/ViewSwitcher";
import FirePanel from "@/components/FirePanel";
import RiskPanel, { RiskLayers } from "@/components/RiskPanel";
import Header from "@/components/Header";
import CurrentSummary from "@/components/CurrentSummary";
import type { CurrentData } from "@/lib/fires";

// MapLibre needs to be client-side only (no SSR)
const WildfireMap = dynamic(() => import("@/components/WildfireMap"), {
  ssr: false,
  loading: () => (
    <div className="flex-1 flex items-center justify-center bg-gray-900">
      <div className="text-gray-400">Loading map…</div>
    </div>
  ),
});

export type SelectedFire = {
  fire_id: string;
  fire_name?: string;
  lat: number;
  lon: number;
  size_ha: number;
  fwi?: number;
  isi?: number;
  wind_speed_ms?: number;
  wind_dir_deg?: number;
  source: "active" | "hotspot" | "history";
  properties: Record<string, unknown>;
};

export default function Home() {
  const [view, setView] = useState<View>("current");
  const [historyYear, setHistoryYear] = useState(2023);
  const [riskMonth, setRiskMonth] = useState(new Date().getMonth() + 1);
  const [selectedFire, setSelectedFire] = useState<SelectedFire | null>(null);
  const [currentData, setCurrentData] = useState<CurrentData | null>(null);
  const [riskLayers, setRiskLayers] = useState<RiskLayers>({ lightning: true, human: true });

  return (
    <div className="flex flex-col h-dvh overflow-hidden">
      <Header />
      {/* Phones: map on top, sidebar below. Wider screens: sidebar on the left */}
      <div className="flex flex-col md:flex-row flex-1 min-h-0">
        {/* Sidebar */}
        <aside className="order-2 md:order-1 h-[45%] md:h-auto w-full md:w-80 bg-gray-900 border-t md:border-t-0 md:border-r border-gray-800 flex flex-col">
          <ViewSwitcher
            current={view}
            onChange={(v) => { setView(v); setSelectedFire(null); }}
            historyYear={historyYear}
            onYearChange={(y) => { setHistoryYear(y); setSelectedFire(null); }}
            riskMonth={riskMonth}
            onRiskMonthChange={setRiskMonth}
          />
          <div className="flex-1 overflow-y-auto sidebar-scroll">
            {view === "risk" ? (
              <RiskPanel month={riskMonth} layers={riskLayers} onLayerToggle={setRiskLayers} />
            ) : !selectedFire && view === "current" ? (
              <CurrentSummary data={currentData} onSelect={setSelectedFire} />
            ) : (
              <FirePanel fire={selectedFire} hotspots={currentData?.hotspots ?? []} onBack={() => setSelectedFire(null)} />
            )}
          </div>
        </aside>

        {/* Map */}
        <main className="order-1 md:order-2 flex-1 min-h-0 relative">
          <WildfireMap
            view={view}
            onCurrentData={setCurrentData}
            historyYear={historyYear}
            riskMonth={riskMonth}
            riskLayers={riskLayers}
            onFireSelect={setSelectedFire}
            selectedFire={selectedFire}
          />
        </main>
      </div>
    </div>
  );
}
