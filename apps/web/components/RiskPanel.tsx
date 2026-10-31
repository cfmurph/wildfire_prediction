"use client";

import { Zap, Car } from "lucide-react";
import { cn } from "@/lib/utils";

const MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"];

export type RiskLayers = { lightning: boolean; human: boolean };

interface Props {
  month: number;
  layers: RiskLayers;
  onLayerToggle: (layers: RiskLayers) => void;
}

/**
 * The ignition model's absolute probabilities are not calibrated, so this view
 * only says where fires are more or less likely to start within a month.
 */
export default function RiskPanel({ month, layers, onLayerToggle }: Props) {
  const name = MONTH_NAMES[month - 1];
  const toggles = [
    { key: "lightning", label: "Lightning-caused starts", icon: Zap, on: "bg-blue-900/40 text-blue-300 border border-blue-700", dot: "bg-blue-400" },
    { key: "human",     label: "Human-caused starts",     icon: Car, on: "bg-amber-900/40 text-amber-300 border border-amber-700", dot: "bg-amber-400" },
  ] as const;

  return (
    <div className="p-4 space-y-4 text-sm">
      <div>
        <h2 className="font-semibold text-white text-base">{name}: where fires tend to start</h2>
        <p className="text-xs text-gray-400 mt-1 leading-relaxed">
          Bigger circles mark places where a fire is more likely to start in a typical {name}, compared with the
          rest of BC. The ranking comes from past fire starts, lightning, human activity and the month&apos;s usual
          fire weather.
        </p>
      </div>

      <p className="text-xs text-gray-500 bg-gray-800 rounded-lg p-3 leading-relaxed">
        This is a relative ranking, not a probability and not today&apos;s fire danger. For current conditions,
        see the Current tab or the BC Wildfire Service danger rating.
      </p>

      <div className="space-y-1.5">
        {toggles.map(({ key, label, icon: Icon, on, dot }) => (
          <button
            key={key}
            onClick={() => onLayerToggle({ ...layers, [key]: !layers[key] })}
            className={cn(
              "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors",
              layers[key] ? on : "bg-gray-800 text-gray-500",
            )}
          >
            <Icon className="w-3.5 h-3.5" />
            <span>{label}</span>
            {layers[key] && <span className={cn("ml-auto w-2 h-2 rounded-full", dot)} />}
          </button>
        ))}
      </div>
    </div>
  );
}
