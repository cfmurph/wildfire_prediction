"use client";

import { MAP_VIEWS, type ViewId } from "@/lib/views";

type ViewSwitcherProps = {
  active: ViewId;
  onChange: (id: ViewId) => void;
};

export default function ViewSwitcher({ active, onChange }: ViewSwitcherProps) {
  return (
    <nav className="view-switcher" role="tablist" aria-label="Map views">
      {MAP_VIEWS.map((view) => {
        const selected = view.id === active;
        return (
          <button
            key={view.id}
            type="button"
            className="view-tab"
            role="tab"
            id={`view-tab-${view.id}`}
            aria-selected={selected}
            aria-controls="map-panel"
            onClick={() => onChange(view.id)}
          >
            {view.label}
            {view.available ? null : <span className="soon-badge">Soon</span>}
          </button>
        );
      })}
    </nav>
  );
}
