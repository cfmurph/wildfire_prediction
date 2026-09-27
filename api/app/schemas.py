"""Request and response models for the wildfire API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClusterIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, max_length=80)
    latitude: float = Field(ge=47, le=62)
    longitude: float = Field(ge=-142, le=-110)
    hotspot_count: int = Field(ge=1, le=100_000)
    max_frp: float | None = Field(default=None, ge=0, le=1_000_000)
    mean_frp: float | None = Field(default=None, ge=0, le=1_000_000)
    max_brightness: float | None = Field(default=None, ge=0, le=1000)
    latest_acquisition: str | None = Field(default=None, max_length=40)
    confidence: dict[str, int] = Field(default_factory=dict)
    satellites: list[str] = Field(default_factory=list)
    daynight: dict[str, int] = Field(default_factory=dict)
    bbox: list[float] | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def limit_confidence(cls, value: object) -> dict[str, int]:
        # Run before the dict[str, int] check so a bad count becomes 0
        # instead of a 422.
        incoming = value if isinstance(value, dict) else {}
        clean: dict[str, int] = {}
        for key in ("high", "nominal", "low", "unknown"):
            raw = incoming.get(key, 0)
            try:
                clean[key] = max(0, min(int(raw), 100_000))
            except (TypeError, ValueError):
                clean[key] = 0
        return clean

    @field_validator("satellites", mode="before")
    @classmethod
    def limit_satellites(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value[:8]:
            token = "".join(ch for ch in str(item) if ch.isalnum() or ch in "-_")[:12]
            if token:
                cleaned.append(token)
        return cleaned


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cluster: ClusterIn
