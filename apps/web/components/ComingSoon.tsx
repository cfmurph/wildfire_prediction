"use client";

import type { MapView } from "@/lib/views";

type ComingSoonProps = {
  view: MapView;
};

export default function ComingSoon({ view }: ComingSoonProps) {
  return (
    <aside className="panel" id="map-panel" aria-labelledby={`view-tab-${view.id}`}>
      <div className="panel-block">
        <p className="eyebrow">Coming soon</p>
        <h2>{view.label}</h2>
        <p className="lede">{view.summary}</p>
        <div className="notice" role="status">
          <strong>This view is not in the current release.</strong>
          <p>
            The map stays in place. This tab will gain its own layer
            {view.id === "history" ? " and a 2012–2023 time slider" : ""} when that data is connected.
          </p>
        </div>
      </div>
      <p className="disclaimer">
        Current wildfires are the live view. History, long-term risk, and short-term spread are
        planned on the same map.
      </p>
    </aside>
  );
}
