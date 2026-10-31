import { FIRE_STATUSES, HISTORY_SIZES } from "@/lib/utils";

function Dot({ color, ring = false, size = 10 }: { color: string; ring?: boolean; size?: number }) {
  return (
    <span
      className="inline-block rounded-full shrink-0"
      style={ring ? { width: size, height: size, border: `2px solid ${color}` } : { width: size, height: size, background: color }}
    />
  );
}

const BOX = "absolute top-3 left-3 bg-gray-900/90 border border-gray-700 rounded-lg p-3 text-xs text-gray-300 space-y-1.5 pointer-events-none";

/** Key for the Current and Predictions map symbols. */
export default function MapLegend({ showSpread }: { showSpread: boolean }) {
  return (
    // Hidden on phones, where it would cover most of the map; the sidebar repeats the status colours
    <div className={`hidden sm:block ${BOX}`}>
      <p className="text-[10px] uppercase tracking-wide text-gray-500">Active fires (size = area)</p>
      {FIRE_STATUSES.map(({ status, color }) => (
        <div key={status} className="flex items-center gap-2"><Dot color={color} />{status}</div>
      ))}
      <div className="flex items-center gap-2 pt-1"><Dot color="#fde68a" size={6} />Satellite hotspot, last 24 h</div>
      <div className="flex items-center gap-2"><Dot color="#eab308" ring />Weather station (colour = FWI)</div>
      {showSpread && (
        <>
          <p className="text-[10px] uppercase tracking-wide text-gray-500 pt-1">24 h spread (heuristic)</p>
          <div className="flex items-center gap-2"><Dot color="#ef4444" />Conservative (p25)</div>
          <div className="flex items-center gap-2"><Dot color="#f97316" />Median (p50)</div>
          <div className="flex items-center gap-2"><Dot color="#fca5a5" />Upper bound (p75)</div>
        </>
      )}
    </div>
  );
}

/** null while loading, "error" when the perimeters could not be fetched. */
export type HistorySummary = { year: number; count: number; ha: number } | "error" | null;

/** Year totals and the perimeter colour key for the History view. */
export function HistoryLegend({ summary }: { summary: HistorySummary }) {
  return (
    <div className={BOX}>
      {summary === "error" ? (
        <p className="text-red-300">Fire perimeters are unavailable right now.</p>
      ) : summary ? (
        <p className="text-white font-semibold">
          {summary.year}: {summary.count.toLocaleString()} fire{summary.count === 1 ? "" : "s"} · {Math.round(summary.ha).toLocaleString()} ha
        </p>
      ) : (
        <p className="text-gray-500">Loading perimeters…</p>
      )}
      {/* Only the totals on phones */}
      <div className="hidden sm:block space-y-1.5">
        <p className="text-[10px] uppercase tracking-wide text-gray-500 pt-1">Burned area, by final fire size</p>
        {HISTORY_SIZES.map(({ label, color }) => (
          <div key={label} className="flex items-center gap-2"><Dot color={color} />{label}</div>
        ))}
      </div>
    </div>
  );
}
