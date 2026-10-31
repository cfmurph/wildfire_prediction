"use client";

import { useEffect, useState } from "react";
import { Flame } from "lucide-react";
import { cn } from "@/lib/utils";

const API = process.env.NEXT_PUBLIC_API_URL;

export default function Header() {
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(`${API}/health`).then((r) => setApiUp(r.ok)).catch(() => setApiUp(false));
  }, []);

  return (
    <header className="h-12 bg-gray-900 border-b border-gray-800 flex items-center px-4 gap-3 shrink-0">
      <Flame className="text-orange-500 w-5 h-5" />
      <span className="font-semibold text-sm tracking-wide">BC Wildfire Prediction</span>
      <span className="text-xs text-gray-500 ml-1">— Research prototype, not an official forecast</span>
      <div className="ml-auto flex items-center gap-2 text-xs text-gray-500">
        <span className={cn("w-2 h-2 rounded-full inline-block", apiUp === null ? "bg-gray-500" : apiUp ? "bg-green-500" : "bg-red-500")} />
        {apiUp === false ? "Data offline" : "Live"}
      </div>
    </header>
  );
}
