"""FastAPI backend for Stage 4: a thin HTTP wrapper around the Stage 3
`care_facilities.pipeline` entrypoint, plus a static single-page frontend.

Endpoints:
    GET  /health         -- liveness check (unguarded; used by the host).
    POST /search         -- {zipcode, radius_miles} -> ranked facility list.
    POST /search/stream  -- same search as SSE (progress events + final result).
    GET  /               -- serves api/static/index.html (and other static assets).

When `APP_USERNAME` and `APP_PASSWORD` are both set, every request except
`/health` is gated behind HTTP Basic Auth. Rate limiting on `/search*`
caps abuse of the outbound CMS/NPPES/Census calls.

`pipeline.run` / `run_detailed` perform real, potentially slow (cold-cache)
network calls, so they always run in a worker thread
(`starlette.concurrency.run_in_threadpool`) rather than on the FastAPI
event loop.

Honesty guardrail: this module never invents facility data -- it only
serializes whatever `Facility` objects the pipeline returns.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import secrets
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from care_facilities import config
from care_facilities import pipeline

_ZIPCODE_RE = re.compile(r"^\d{5}$")

# Generous enough for one real user; tight enough to stop casual scraping.
_SEARCH_RATE_LIMIT = "20/hour"


class SearchRequest(BaseModel):
    zipcode: str
    radius_miles: float | None = Field(default=None, gt=0)

    @field_validator("zipcode")
    @classmethod
    def validate_zipcode(cls, value: str) -> str:
        value = (value or "").strip()
        if not _ZIPCODE_RE.match(value):
            raise ValueError(
                f"zipcode must be a 5-digit US zip code (e.g. '94404'); got {value!r}"
            )
        return value


def _auth_enabled() -> bool:
    return bool(config.APP_USERNAME and config.APP_PASSWORD)


def _credentials_valid(username: str, password: str) -> bool:
    if not _auth_enabled():
        return True
    user_ok = secrets.compare_digest(username, config.APP_USERNAME or "")
    pass_ok = secrets.compare_digest(password, config.APP_PASSWORD or "")
    return user_ok and pass_ok


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Gate every path except /health when APP_USERNAME + APP_PASSWORD are set.

    Applied as middleware (rather than a FastAPI Depends) so the StaticFiles
    mount at "/" is covered too -- a Depends on route handlers wouldn't be.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health" or not _auth_enabled():
            return await call_next(request)

        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth[6:]).decode("utf-8")
                username, _, password = decoded.partition(":")
                if _credentials_valid(username, password):
                    return await call_next(request)
            except Exception:  # noqa: BLE001 - treat any decode failure as 401
                pass

        return Response(
            content="Authentication required",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Care Facilities"'},
            media_type="text/plain",
        )


def _serialize_search_payload(
    zipcode: str, radius: float, results: list, errors: list[str]
) -> dict[str, Any]:
    return {
        "zipcode": zipcode,
        "radius_miles": radius,
        "count": len(results),
        "results": [facility.model_dump() for facility in results],
        "errors": errors,
    }


limiter = Limiter(key_func=get_remote_address, default_limits=[])

app = FastAPI(title="Care Facilities Finder API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Auth before CORS so unauthenticated responses aren't CORS-preflight noise.
app.add_middleware(BasicAuthMiddleware)

# Same-origin UI is the normal path; keep CORS open for local experimentation.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
@limiter.limit(_SEARCH_RATE_LIMIT)
async def search(request: Request, body: SearchRequest) -> dict:
    radius = (
        body.radius_miles
        if body.radius_miles is not None
        else float(config.DEFAULT_RADIUS_MILES)
    )

    try:
        detailed = await run_in_threadpool(
            pipeline.run_detailed, body.zipcode, radius
        )
    except Exception:  # noqa: BLE001 - never let this become a raw 500 traceback
        return _serialize_search_payload(
            body.zipcode, radius, [], ["Search failed. Please try again."]
        )

    return _serialize_search_payload(
        body.zipcode,
        detailed["radius_miles"],
        detailed["results"],
        detailed["errors"],
    )


@app.post("/search/stream")
@limiter.limit(_SEARCH_RATE_LIMIT)
async def search_stream(request: Request, body: SearchRequest) -> StreamingResponse:
    """SSE stream: progress events, then a final `done` (or `error`) event.

    Event payloads are JSON objects with at least ``stage`` and ``message``.
    The terminal event is ``stage=done`` with the same fields as ``POST /search``,
    or ``stage=error`` if the pipeline itself crashed.
    """
    radius = (
        body.radius_miles
        if body.radius_miles is not None
        else float(config.DEFAULT_RADIUS_MILES)
    )
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def on_progress(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _run() -> dict[str, Any]:
        return pipeline.run_detailed(body.zipcode, radius, on_progress=on_progress)

    async def event_gen():
        task = asyncio.create_task(run_in_threadpool(_run))
        # Drain progress while the worker runs.
        while not task.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # Keepalive comment so proxies don't idle-close the stream.
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"

        try:
            detailed = await task
        except Exception:  # noqa: BLE001
            fail = {
                "stage": "error",
                "message": "Search failed. Please try again.",
                **_serialize_search_payload(
                    body.zipcode, radius, [], ["Search failed. Please try again."]
                ),
            }
            yield f"data: {json.dumps(fail)}\n\n"
            return

        done = {
            "stage": "done",
            "message": f"Found {len(detailed['results'])} facilit(ies).",
            **_serialize_search_payload(
                body.zipcode,
                detailed["radius_miles"],
                detailed["results"],
                detailed["errors"],
            ),
        }
        yield f"data: {json.dumps(done)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Mounted last so it doesn't shadow the routes defined above; `html=True`
# means "/" serves static/index.html.
app.mount(
    "/",
    StaticFiles(directory="api/static", html=True),
    name="static",
)
