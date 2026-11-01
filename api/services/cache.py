"""Short-lived cache for upstream feeds, so visitor traffic does not reach BCWS, FIRMS or CWFIS."""

from __future__ import annotations

import functools
import time

_caches: list[dict] = []


def ttl_cache(seconds: float):
    """Cache an async function's GeoJSON result per arguments. Failures and empty results are not kept."""
    def decorate(fn):
        store: dict = {}
        _caches.append(store)

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            hit = store.get(key)
            if hit and time.monotonic() - hit[0] < seconds:
                return hit[1]
            value = await fn(*args, **kwargs)
            if value and value.get("features"):
                store[key] = (time.monotonic(), value)
            return value

        return wrapper
    return decorate


def clear_all() -> None:
    for store in _caches:
        store.clear()
