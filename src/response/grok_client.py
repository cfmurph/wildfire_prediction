"""
xAI Grok Decision-Support Client
==================================
Wraps the xAI Grok API (OpenAI-compatible) to generate:

  1. Natural language situation reports for emergency managers
  2. Evacuation zone priority recommendations
  3. Conversational answers about fire state

Grok is positioned DOWNSTREAM of all ML models. It never generates
spatial predictions — those come from ConvLSTM/U-Net with calibrated
uncertainty. Grok receives a structured JSON payload and produces
plain-English outputs.

Usage:
  from src.response.grok_client import GrokClient, build_fire_payload
  client = GrokClient()
  report = client.situation_report(payload)

Configuration:
  Set XAI_API_KEY environment variable or pass api_key to GrokClient.
  Model and prompt paths are configured in configs/grok.yaml.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured payload schema
# ---------------------------------------------------------------------------

@dataclass
class SpreadProbabilities:
    """Percentile-based spread area forecasts from the ML model."""
    p25_area_ha: float   # 25th percentile of ensemble spread
    p50_area_ha: float   # median
    p75_area_ha: float   # 75th percentile


@dataclass
class FWISummary:
    """Current Fire Weather Index system values."""
    FWI: float    # Fire Weather Index (operational severity scale)
    ISI: float    # Initial Spread Index (rate-of-spread potential)
    BUI: float    # Build-up Index (fuel availability)
    FFMC: float   # Fine Fuel Moisture Code


@dataclass
class WindSummary:
    speed_kmh: float
    direction_deg: float
    direction_label: str  # "NW", "SE", etc.


@dataclass
class RiskZone:
    """A named geographic zone with assessed burn risk."""
    name: str
    burn_probability: float       # from ML model
    population: int
    road_access: str              # "open" | "limited" | "none"
    distance_to_fire_km: float


@dataclass
class FirePayload:
    """
    Structured JSON payload passed to Grok.
    Assembled from ML model outputs + weather + infrastructure data.
    """
    fire_id: str
    fire_name: Optional[str]
    province: str
    as_of_datetime: str           # ISO-8601
    current_area_ha: float
    spread_probabilities: SpreadProbabilities
    fwi_summary: FWISummary
    wind_summary: WindSummary
    high_risk_zones: list[RiskZone]
    # BurnP3+ physics prior (if available)
    burnp3_burn_prob_max: Optional[float] = None
    burnp3_available: bool = False


def build_fire_payload(
    fire_id: str,
    as_of_datetime: str,
    current_area_ha: float,
    ml_spread_probs: dict,       # {"p25": ..., "p50": ..., "p75": ...}
    fwi_values: dict,
    wind_values: dict,
    risk_zones: list[dict],
    fire_name: Optional[str] = None,
    province: str = "BC",
    burnp3_max: Optional[float] = None,
) -> FirePayload:
    """Convenience constructor from raw dicts."""
    return FirePayload(
        fire_id=fire_id,
        fire_name=fire_name,
        province=province,
        as_of_datetime=as_of_datetime,
        current_area_ha=current_area_ha,
        spread_probabilities=SpreadProbabilities(
            p25_area_ha=ml_spread_probs.get("p25", 0.0),
            p50_area_ha=ml_spread_probs.get("p50", 0.0),
            p75_area_ha=ml_spread_probs.get("p75", 0.0),
        ),
        fwi_summary=FWISummary(
            FWI=fwi_values.get("FWI", 0.0),
            ISI=fwi_values.get("ISI", 0.0),
            BUI=fwi_values.get("BUI", 0.0),
            FFMC=fwi_values.get("FFMC", 0.0),
        ),
        wind_summary=WindSummary(
            speed_kmh=wind_values.get("speed_kmh", 0.0),
            direction_deg=wind_values.get("direction_deg", 0.0),
            direction_label=wind_values.get("direction_label", ""),
        ),
        high_risk_zones=[
            RiskZone(
                name=z["name"],
                burn_probability=z.get("burn_probability", 0.0),
                population=z.get("population", 0),
                road_access=z.get("road_access", "open"),
                distance_to_fire_km=z.get("distance_to_fire_km", 0.0),
            )
            for z in risk_zones
        ],
        burnp3_burn_prob_max=burnp3_max,
        burnp3_available=burnp3_max is not None,
    )


def _degrees_to_compass(deg: float) -> str:
    """Convert wind direction in degrees to compass label."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(deg / 45) % 8
    return directions[idx]


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def _load_prompt(prompt_path: str) -> str:
    p = Path(prompt_path)
    if p.exists():
        return p.read_text().strip()
    log.warning(f"Prompt file not found: {prompt_path}. Using built-in default.")
    return _DEFAULT_SITUATION_REPORT_PROMPT


_DEFAULT_SITUATION_REPORT_PROMPT = """You are a wildfire situation analyst for the BC Wildfire Service.
Given the following structured fire intelligence report, produce a concise natural-language
situation briefing (4–6 sentences) for an emergency manager. Be specific about numbers and
locations. Flag the highest-priority evacuation zone. Note confidence level based on FWI.

Fire intelligence data:
{payload_json}

Situation briefing:"""


# ---------------------------------------------------------------------------
# Grok client
# ---------------------------------------------------------------------------

class GrokClient:
    """
    xAI Grok API client for wildfire decision support.

    The xAI API is OpenAI-compatible — uses the openai Python package
    pointed at https://api.x.ai/v1.

    Parameters
    ----------
    api_key : str, optional
        xAI API key. Defaults to XAI_API_KEY environment variable.
    model : str
        Grok model slug. "grok-3" for highest quality,
        "grok-3-mini" for faster/cheaper drafts.
    max_tokens : int
    temperature : float
        Low (0.1–0.3) for factual situation reports.
    situation_report_prompt : str
        Path to the prompt template file.
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Number of retries on transient failures.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-3",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        situation_report_prompt: str = "prompts/situation_report.txt",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._situation_prompt_template = _load_prompt(situation_report_prompt)

        resolved_key = api_key or os.environ.get("XAI_API_KEY", "")
        if not resolved_key:
            log.warning(
                "XAI_API_KEY not set. Grok calls will fail. "
                "Export XAI_API_KEY=your_key or pass api_key to GrokClient()."
            )

        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=resolved_key,
                base_url="https://api.x.ai/v1",
                timeout=timeout,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def situation_report(self, payload: FirePayload) -> str:
        """
        Generate a natural-language wildfire situation report.

        Parameters
        ----------
        payload : FirePayload
            Structured fire intelligence assembled from ML model outputs.

        Returns
        -------
        str — Plain-English situation briefing (4–6 sentences).
        """
        payload_json = json.dumps(self._serialise_payload(payload), indent=2)
        prompt = self._situation_prompt_template.format(payload_json=payload_json)

        return self._chat(prompt)

    def evacuation_recommendation(self, payload: FirePayload) -> str:
        """
        Generate ranked evacuation zone priorities with justification.

        Returns a structured plain-English recommendation listing zones
        in priority order with burn probability and road access.
        """
        zones_json = json.dumps(
            [
                {
                    "name": z.name,
                    "burn_probability": z.burn_probability,
                    "population": z.population,
                    "road_access": z.road_access,
                    "distance_km": z.distance_to_fire_km,
                }
                for z in payload.high_risk_zones
            ],
            indent=2,
        )

        prompt = f"""You are a wildfire evacuation planner for the BC Wildfire Service.
Fire: {payload.fire_id} ({payload.fire_name or 'unnamed'}) — {payload.current_area_ha:,.0f} ha as of {payload.as_of_datetime}
Wind: {payload.wind_summary.speed_kmh:.0f} km/h from {payload.wind_summary.direction_label}
FWI: {payload.fwi_summary.FWI:.0f} (ISI={payload.fwi_summary.ISI:.0f})
ML median spread forecast: {payload.spread_probabilities.p50_area_ha:,.0f} ha in 24h
(75th pct: {payload.spread_probabilities.p75_area_ha:,.0f} ha)

At-risk zones:
{zones_json}

Rank these zones in evacuation priority order. For each zone state:
1. Priority ranking
2. Key reason (burn probability + road access + population)
3. Recommended action (Evacuation Order / Evacuation Alert / Monitor)
Be concise and specific."""

        return self._chat(prompt)

    def fire_update_query(self, payload: FirePayload, query: str) -> str:
        """
        Answer an operator's natural-language query about the current fire state.

        Parameters
        ----------
        payload : FirePayload
        query : str — e.g. "What changed in the last 6 hours on this fire?"
        """
        context = json.dumps(self._serialise_payload(payload), indent=2)
        prompt = f"""You are a wildfire intelligence analyst.
Current fire intelligence data:
{context}

Operator query: {query}

Answer concisely and factually based only on the data provided above."""

        return self._chat(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(self, prompt: str) -> str:
        """Send a prompt to the Grok API with retry logic."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_exc = exc
                log.warning(f"Grok API attempt {attempt}/{self.max_retries} failed: {exc}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)   # exponential backoff

        raise RuntimeError(f"Grok API failed after {self.max_retries} attempts: {last_exc}")

    def _serialise_payload(self, payload: FirePayload) -> dict:
        """Convert FirePayload dataclass to a JSON-serialisable dict."""
        d = asdict(payload)
        # Ensure wind direction label is included
        d["wind_summary"]["direction_label"] = _degrees_to_compass(
            payload.wind_summary.direction_deg
        )
        return d
