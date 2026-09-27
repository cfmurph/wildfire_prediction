"""Shared response for map views that are not implemented yet."""

from __future__ import annotations

from fastapi.responses import JSONResponse

# Product order: history, long-term risk, current conditions, short-term spread.
PLANNED_VIEWS: dict[str, str] = {
    "history": "Past BC fires from 2012–2023, with a time slider.",
    "risk": "Long-term burn likelihood across British Columbia.",
    "predict": "Next-day spread for each active fire.",
}


def coming_soon(view: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "view": view,
            "status": "coming_soon",
            "detail": PLANNED_VIEWS[view],
        },
    )
