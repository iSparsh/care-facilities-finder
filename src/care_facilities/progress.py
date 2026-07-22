"""Optional progress reporting for long-running pipeline searches.

Callers (e.g. the SSE search endpoint) install a callback via
`progress_scope`; pipeline / source code calls `emit(...)` at stage
boundaries and during slow loops (ALF geocoding). If no callback is
installed, `emit` is a no-op -- library/CLI use is unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator

ProgressCallback = Callable[[dict[str, Any]], None]

_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "care_facilities_progress", default=None
)


@contextmanager
def progress_scope(callback: ProgressCallback | None) -> Iterator[None]:
    """Install `callback` for the duration of the with-block (and child work)."""
    token = _callback.set(callback)
    try:
        yield
    finally:
        _callback.reset(token)


def emit(stage: str, message: str, **extra: Any) -> None:
    """Send a progress event to the active callback, if any."""
    cb = _callback.get()
    if cb is None:
        return
    event: dict[str, Any] = {"stage": stage, "message": message}
    event.update(extra)
    try:
        cb(event)
    except Exception:  # noqa: BLE001 - progress must never break the search
        pass
