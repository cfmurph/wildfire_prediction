import { FIRE_STATUSES } from "@/lib/utils";

function Dot({ color, ring = false, size = 10 }: { color: string; ring?: boolean; size?: number }) {
  return (
    <span
      className="inline-block rounded-full shrink-0"
      style={ring ? { width: size, height: size, border: `2px solid ${color}` } : { width: size, height: size, background: color }}
    />
  );
}

/** Key for the Current and Predictions map symbols. */
export default function MapLegend({ showSpread }: { showSpread: boolean }) {
  return (
    <div className="absolute top-3 left-3 bg-gray-900/90 border border-gray-700 rounded-lg p-3 text-xs text-gray-300 space-y-1.5 pointer-events-none">
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
