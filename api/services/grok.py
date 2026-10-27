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
    fwi: float,
    isi: float,
    wind_speed_ms: float,
    wind_dir_deg: float,
    spread_p50_ha: float,
    spread_p75_ha: float,
) -> str:
    """
    Generate a plain-English situation report via xAI Grok.
    Returns a string even on failure (graceful degradation).
    """
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        return (
            f"**{fire_name or fire_id}** — {current_area_ha:,.0f} ha\n\n"
            f"Fire Weather Index: {fwi:.0f} | Wind: {wind_speed_ms:.0f} m/s\n"
            f"Median 24h spread forecast: {spread_p50_ha:,.0f} ha (upper bound: {spread_p75_ha:,.0f} ha)\n\n"
            "_Grok situation report unavailable — set XAI_API_KEY for AI-generated briefings._"
        )

    payload_json = json.dumps(
        {
            "fire_id": fire_id,
            "fire_name": fire_name or fire_id,
            "province": "BC",
            "location": {"lat": lat, "lon": lon},
            "current_area_ha": current_area_ha,
            "spread_forecast_24h": {
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
Be specific about numbers. State the main risk driver (wind/drought/fuel). Recommend one immediate action.

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
        return (
            f"**{fire_name or fire_id}** — {current_area_ha:,.0f} ha\n\n"
            f"FWI: {fwi:.0f} | ISI: {isi:.0f} | Wind: {wind_speed_ms:.0f} m/s\n"
            f"24h spread forecast: {spread_p50_ha:,.0f}–{spread_p75_ha:,.0f} ha\n\n"
            "_AI briefing temporarily unavailable._"
        )
