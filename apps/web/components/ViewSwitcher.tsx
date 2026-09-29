"use client";

import { Flame, Clock, TrendingUp, Map } from "lucide-react";
import { cn } from "@/lib/utils";

export type View = "current" | "history" | "predictions" | "risk";

const VIEWS: { id: View; label: string; icon: React.ElementType; description: string }[] = [
  { id: "current",     label: "Current",     icon: Flame,      description: "Live hotspots + FWI" },
  { id: "history",     label: "History",     icon: Clock,      description: "BC fires 2006–2024" },
  { id: "predictions", label: "Predictions", icon: TrendingUp, description: "Next-day spread" },
  { id: "risk",        label: "Risk",        icon: Map,        description: "12-month forecast" },
];

const MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

interface Props {
  current: View;
  onChange: (v: View) => void;
  historyYear: number;
  onYearChange: (y: number) => void;
  riskMonth: number;
  onRiskMonthChange: (m: number) => void;
}

export default function ViewSwitcher({ current, onChange, historyYear, onYearChange, riskMonth, onRiskMonthChange }: Props) {
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

      {/* Month slider + layer toggle — Risk view */}
      {current === "risk" && (
        <div className="pt-1">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>12-month forecast</span>
            <span className="font-semibold text-orange-400">{MONTH_NAMES[riskMonth - 1]} 2027</span>
          </div>
          <input
            type="range"
            min={1}
            max={12}
            value={riskMonth}
            onChange={(e) => onRiskMonthChange(Number(e.target.value))}
            className="w-full accent-orange-500 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-gray-600 mt-0.5">
            <span>Jan</span>
            <span className="text-orange-500">🔥 Jul-Aug</span>
            <span>Dec</span>
          </div>
          <div className="mt-1.5 grid grid-cols-12 gap-px">
            {MONTH_NAMES.map((m, i) => (
              <button
                key={m}
                onClick={() => onRiskMonthChange(i + 1)}
                className={`text-[8px] py-0.5 rounded transition-colors ${
                  riskMonth === i + 1
                    ? "bg-orange-600 text-white"
                    : [5,6,7,8,9].includes(i + 1)
                    ? "bg-orange-900/40 text-orange-300 hover:bg-orange-800/40"
                    : "bg-gray-800 text-gray-500 hover:bg-gray-700"
                }`}
              >
                {m[0]}
              </button>
            ))}
          </div>
        </div>
      )}

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
