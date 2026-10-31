"""
Grok situation report service (thin async wrapper around the existing client).
"""

from __future__ import annotations

import json
import os
import logging
from typing import Optional

log = logging.getLogger(__name__)


async def generate_situation_report(
    fire_id: str,
    fire_name: Optional[str],
    current_area_ha: float,
    lat: float,
    lon: float,
    fwi: Optional[float],
    isi: Optional[float],
    wind_speed_ms: Optional[float],
    wind_dir_deg: Optional[float],
    spread_p50_ha: Optional[float],
    spread_p75_ha: Optional[float],
) -> str:
    """
    Generate a plain-English situation report via xAI Grok.
    Returns a string even on failure (graceful degradation).
    """
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        return _summary(fire_id, fire_name, current_area_ha, fwi, isi, wind_speed_ms, spread_p50_ha, spread_p75_ha) + (
            "_Grok situation report unavailable — set XAI_API_KEY for AI-generated briefings._"
        )

    payload_json = json.dumps(
        {
            "fire_id": fire_id,
            "fire_name": fire_name or fire_id,
            "province": "BC",
            "location": {"lat": lat, "lon": lon},
            "current_area_ha": current_area_ha,
            "heuristic_spread_estimate_24h": {
                "p50_area_ha": spread_p50_ha,
                "p75_area_ha": spread_p75_ha,
            },
            "fwi_summary": {"FWI": fwi, "ISI": isi},
            "wind": {"speed_ms": wind_speed_ms, "direction_deg": wind_dir_deg},
        },
        indent=2,
    )

    prompt = f"""You are a wildfire situation analyst for the BC Wildfire Service.

Given this fire intelligence data, write a concise 3-4 sentence situation briefing for an emergency manager.
Be specific about numbers. Weather comes from the nearest CWFIS station; if it is null, say no current weather is available and do not guess. The spread figures are a rough wind/FWI heuristic, not a model forecast; say so if you use them. State the main risk driver (wind/drought/fuel). Recommend one immediate action.

Fire data:
{payload_json}

Situation briefing:"""

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        response = await client.chat.completions.create(
            model="grok-3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        log.error("Grok API error (%s)", type(exc).__name__)
        return _summary(fire_id, fire_name, current_area_ha, fwi, isi, wind_speed_ms, spread_p50_ha, spread_p75_ha) + (
            "_AI briefing temporarily unavailable._"
        )


def _summary(fire_id, fire_name, area_ha, fwi, isi, wind_speed_ms, p50_ha, p75_ha) -> str:
    """Plain summary used when Grok is unavailable."""
    if fwi is None:
        weather = "No CWFIS weather station is reporting near this fire, so there is no spread estimate.\n\n"
    else:
        weather = (
            f"FWI: {fwi:.0f} | ISI: {isi:.0f} | Wind: {wind_speed_ms:.0f} m/s (nearest CWFIS station)\n"
            f"Heuristic 24h spread estimate (not an ML forecast): {p50_ha:,.0f}–{p75_ha:,.0f} ha\n\n"
        )
    return f"**{fire_name or fire_id}** — {area_ha:,.0f} ha\n\n" + weather
