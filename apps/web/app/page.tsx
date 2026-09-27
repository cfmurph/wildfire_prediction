"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import ViewSwitcher, { View } from "@/components/ViewSwitcher";
import FirePanel from "@/components/FirePanel";
import Header from "@/components/Header";

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
  const [selectedFire, setSelectedFire] = useState<SelectedFire | null>(null);

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 bg-gray-900 border-r border-gray-800 flex flex-col">
          <ViewSwitcher
            current={view}
            onChange={setView}
            historyYear={historyYear}
            onYearChange={setHistoryYear}
          />
          <div className="flex-1 overflow-y-auto sidebar-scroll">
            <FirePanel fire={selectedFire} view={view} />
          </div>
        </aside>

        {/* Map */}
        <main className="flex-1 relative">
          <WildfireMap
            view={view}
            historyYear={historyYear}
            onFireSelect={setSelectedFire}
            selectedFire={selectedFire}
          />
        </main>
      </div>
    </div>
  );
}
