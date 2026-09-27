import { Flame } from "lucide-react";

export default function Header() {
  return (
    <header className="h-12 bg-gray-900 border-b border-gray-800 flex items-center px-4 gap-3 shrink-0">
      <Flame className="text-orange-500 w-5 h-5" />
      <span className="font-semibold text-sm tracking-wide">BC Wildfire Prediction</span>
      <span className="text-xs text-gray-500 ml-1">— Canadian Forest Service · Phase 1</span>
      <div className="ml-auto flex items-center gap-2 text-xs text-gray-500">
        <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
        Live
      </div>
    </header>
  );
}
