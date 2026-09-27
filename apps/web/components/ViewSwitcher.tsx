"use client";

import { Flame, Clock, TrendingUp, Map } from "lucide-react";
import { cn } from "@/lib/utils";

export type View = "current" | "history" | "predictions" | "risk";

const VIEWS: { id: View; label: string; icon: React.ElementType; description: string }[] = [
  { id: "current",     label: "Current",     icon: Flame,      description: "Live hotspots + FWI" },
  { id: "history",     label: "History",     icon: Clock,      description: "BC fires 2006–2024" },
  { id: "predictions", label: "Predictions", icon: TrendingUp, description: "Next-day spread" },
  { id: "risk",        label: "Risk",        icon: Map,        description: "Long-term likelihood" },
];

interface Props {
  current: View;
  onChange: (v: View) => void;
  historyYear: number;
  onYearChange: (y: number) => void;
}

export default function ViewSwitcher({ current, onChange, historyYear, onYearChange }: Props) {
  return (
    <div className="p-3 border-b border-gray-800 space-y-2">
      <div className="grid grid-cols-2 gap-1">
        {VIEWS.map(({ id, label, icon: Icon, description }) => (
          <button
            key={id}
            onClick={() => onChange(id)}
            className={cn(
              "flex flex-col items-start gap-0.5 px-3 py-2 rounded-lg text-left transition-colors",
              current === id
                ? "bg-orange-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200"
            )}
          >
            <div className="flex items-center gap-1.5">
              <Icon className="w-3.5 h-3.5" />
              <span className="text-xs font-medium">{label}</span>
            </div>
            <span className="text-[10px] opacity-70 leading-tight">{description}</span>
          </button>
        ))}
      </div>

      {/* Year slider — only shown for History view */}
      {current === "history" && (
        <div className="pt-1">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Year</span>
            <span className="font-semibold text-orange-400">{historyYear}</span>
          </div>
          <input
            type="range"
            min={2006}
            max={2024}
            value={historyYear}
            onChange={(e) => onYearChange(Number(e.target.value))}
            className="w-full accent-orange-500 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-gray-600 mt-0.5">
            <span>2006</span>
            <span>2017 🔥</span>
            <span>2023 🔥</span>
          </div>
        </div>
      )}
    </div>
  );
}
