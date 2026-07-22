"""FastAPI backend for Stage 4: a thin HTTP wrapper around the Stage 3
`care_facilities.pipeline.run` entrypoint, plus a static single-page
frontend.

Endpoints:
    GET  /health   -- liveness check.
    POST /search   -- {zipcode, radius_miles} -> ranked facility list + metadata.
    GET  /          -- serves api/static/index.html (and other static assets).

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

import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from care_facilities import config
from care_facilities import pipeline

_ZIPCODE_RE = re.compile(r"^\d{5}$")


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


app = FastAPI(title="Care Facilities Finder API")

# Permissive CORS: harmless for this local/dev tool, and lets someone hit the
# API directly from a different origin (e.g. a separately-served frontend
# during development) even though the shipped frontend is same-origin.
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
async def search(request: SearchRequest) -> dict:
    radius = (
        request.radius_miles
        if request.radius_miles is not None
        else float(config.DEFAULT_RADIUS_MILES)
    )

    try:
        results = await run_in_threadpool(pipeline.run, request.zipcode, radius)
    except Exception as exc:  # noqa: BLE001 - never let this become a raw 500 traceback
        return {
            "zipcode": request.zipcode,
            "radius_miles": radius,
            "count": 0,
            "results": [],
            "errors": [f"Search failed: {exc}"],
        }

    return {
        "zipcode": request.zipcode,
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
