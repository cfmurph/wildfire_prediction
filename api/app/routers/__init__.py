"""HTTP routers. Current wildfires are live; history, risk, and predict are seams."""

from app.routers import current, history, predict, risk

__all__ = ["current", "history", "predict", "risk"]
