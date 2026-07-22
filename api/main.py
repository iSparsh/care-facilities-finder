"""FastAPI backend for Stage 4: a thin HTTP wrapper around the Stage 3
`care_facilities.pipeline.run` entrypoint, plus a static single-page
frontend.

Endpoints:
    GET  /health   -- liveness check (unguarded; used by the host).
    POST /search   -- {zipcode, radius_miles} -> ranked facility list + metadata.
    GET  /          -- serves api/static/index.html (and other static assets).

When `APP_USERNAME` and `APP_PASSWORD` are both set, every request except
`/health` is gated behind HTTP Basic Auth. Rate limiting on `/search`
caps abuse of the outbound CMS/NPPES/Census calls.

`pipeline.run` performs real, potentially slow (cold-cache) network calls
(CMS, NPPES, Census geocoder), so it is always executed in a worker thread
(`starlette.concurrency.run_in_threadpool`) rather than directly on the
FastAPI event loop, so the server stays responsive to other requests (e.g.
`/health`) while a search is in flight.

Honesty guardrail carried over from Stage 3: this module never invents or
alters facility data -- it only serializes whatever `Facility` objects
`pipeline.run` returns (via `.model_dump()`), so `None` fields (e.g.
`cms_overall_rating`/`affiliated_aco` for Assisted Living facilities) pass
through as JSON `null`, to be rendered as "N/A" by the frontend -- never
fabricated, never silently dropped.
"""

from __future__ import annotations

import base64
import re
import secrets

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
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
        results = await run_in_threadpool(pipeline.run, body.zipcode, radius)
    except Exception:  # noqa: BLE001 - never let this become a raw 500 traceback
        return {
            "zipcode": body.zipcode,
            "radius_miles": radius,
            "count": 0,
            "results": [],
            "errors": ["Search failed. Please try again."],
        }

    return {
        "zipcode": body.zipcode,
        "radius_miles": radius,
        "count": len(results),
        "results": [facility.model_dump() for facility in results],
        "errors": [],
    }


# Mounted last so it doesn't shadow the routes defined above; `html=True`
# means "/" serves static/index.html.
app.mount(
    "/",
    StaticFiles(directory="api/static", html=True),
    name="static",
)
