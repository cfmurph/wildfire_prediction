"""Process-local TTL cache for upstream hotspot and weather responses."""

from __future__ import annotations

import copy
import threading
import time
from typing import Any


class TTLCache:
    def __init__(self) -> None:
        self._items: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires, value = item
            if expires <= time.monotonic():
                self._items.pop(key, None)
                return None
            return copy.deepcopy(value)

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._items[key] = (time.monotonic() + ttl_seconds, copy.deepcopy(value))

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
