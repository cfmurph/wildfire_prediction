"use client";

import { SelectedFire } from "@/app/page";
import { activeFire, CurrentData } from "@/lib/fires";
import { FIRE_STATUSES, statusColor } from "@/lib/utils";

interface Props {
  data: CurrentData | null;
  onSelect: (fire: SelectedFire) => void;
}

const urgency = (status: unknown) => {
  const i = FIRE_STATUSES.findIndex((s) => s.status === status);
  return i < 0 ? FIRE_STATUSES.length : i;
};

/** Province-wide overview for the Current and Predictions views when no fire is selected. */
export default function CurrentSummary({ data, onSelect }: Props) {
  if (!data) return <p className="p-4 text-sm text-gray-500">Loading fires…</p>;

  const props = (f: GeoJSON.Feature) => (f.properties ?? {}) as Record<string, unknown>;
  // Fires of note first, then most urgent stage of control, then largest
  const fires = [...data.fires].sort((a, b) =>
    Number(props(b).fire_of_note) - Number(props(a).fire_of_note) ||
    urgency(props(a).fire_status) - urgency(props(b).fire_status) ||
    Number(props(b).size_ha) - Number(props(a).size_ha)
  );
  const totalHa = fires.reduce((sum, f) => sum + (Number(props(f).size_ha) || 0), 0);

  return (
    <div className="p-4 space-y-4 text-sm">
      <div>
        <p className="text-white font-semibold">
          {fires.length} active fire{fires.length === 1 ? "" : "s"} · {Math.round(totalHa).toLocaleString()} ha
        </p>
        <p className="text-xs text-gray-500">
          BC Wildfire Service · updated {data.updated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </p>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {FIRE_STATUSES.map(({ status, color }) => (
          <div key={status} className="bg-gray-800 rounded-lg p-2">
            <p className="text-lg font-semibold" style={{ color }}>
              {fires.filter((f) => props(f).fire_status === status).length}
            </p>
            <p className="text-[10px] text-gray-400 leading-tight">{status}</p>
          </div>
        ))}
      </div>

      <div className="text-xs text-gray-400 space-y-0.5">
        <p>{data.hotspots.length.toLocaleString()} satellite hotspots in the last 24 h</p>
        <p>
          {data.stations > 0
            ? `${data.stations} BC weather stations reporting FWI`
            : "No BC weather stations reporting FWI (usual outside the fire season)"}
        </p>
      </div>

      {fires.length === 0 ? (
        <p className="text-gray-500">No active fires reported.</p>
      ) : (
        <ul className="space-y-1">
          {fires.map((f) => {
            const p = props(f);
            return (
              <li key={String(p.fire_number)}>
                <button
                  onClick={() => onSelect(activeFire(f))}
                  className="w-full flex items-start gap-2 text-left rounded-md px-2 py-1.5 hover:bg-gray-800 transition-colors"
                >
                  <span className="mt-1.5 w-2 h-2 rounded-full shrink-0" style={{ background: statusColor(String(p.fire_status)) }} />
                  <span className="flex-1 min-w-0">
                    <span className="block text-gray-200 truncate">
                      {String(p.fire_name || p.fire_number)}{p.fire_of_note ? " · Fire of note" : ""}
                    </span>
                    <span className="block text-[11px] text-gray-500 truncate">{String(p.location || "")}</span>
                  </span>
                  <span className="text-xs text-gray-400 tabular-nums">{Math.round(Number(p.size_ha) || 0).toLocaleString()} ha</span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
