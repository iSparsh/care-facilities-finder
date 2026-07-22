"""LangGraph pipeline: zipcode -> ranked list of `Facility` results.

Pipeline shape (see module-level `build_graph()`):

    resolve_location -> fetch_snf  \\                    /-> reconcile -> rank -> format -> END
                     -> fetch_alf  /-> enrich_alf --------/
                                   \\-> enrich_snf --------/

(fetch_snf/fetch_alf and enrich_snf/enrich_alf each run as independent
parallel branches that fan back in at `reconcile`.)

Every fetch/filter/enrich node is a deterministic wrapper around the
Stage 1/2 `sources.*` functions -- no LLM involved, and each node defends
itself with a try/except so a single flaky HTTP call degrades to an empty
list rather than crashing the whole run.

The one LLM-backed node is `reconcile`, which uses `ChatAnthropic` with
structured output to (a) flag likely duplicate SNF/ALF entries (e.g.
continuing-care communities that show up in both source datasets) and
(b) lightly clean up display strings. If `config.ANTHROPIC_API_KEY` is
unset, or the Anthropic call raises for any reason, `reconcile` falls back
to a plain concatenation of the SNF + ALF facility lists (no dedup, no
cleanup) and appends a warning to `state["errors"]` -- the pipeline is
fully usable without an API key, just less polished.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import aco_match, config, geocode
from .schema import Facility
from .sources import cms_snf, nppes_alf, stengel


class PipelineState(TypedDict, total=False):
    zipcode: str
    radius_miles: float
    origin_lat: float
    origin_lon: float
    state: str
    snf_raw: list[dict]
    alf_raw: list[dict]
    snf_enriched: list[dict]
    alf_enriched: list[dict]
    facilities: list[Facility]
    results: list[Facility]
    errors: list[str]


# --- 1. resolve_location ------------------------------------------------------


def resolve_location(state: PipelineState) -> dict[str, Any]:
    """Resolve `zipcode` to (lat, lon) + 2-letter state. On failure, append
    an error and leave lat/lon/state unset so the routing function below
    short-circuits the run instead of crashing downstream nodes."""
    errors = list(state.get("errors") or [])
    zipcode = state["zipcode"]

    latlng = geocode.zip_to_latlng(zipcode)
    resolved_state = geocode.zip_to_state(zipcode)

    if latlng is None or resolved_state is None:
        errors.append(
            f"Could not resolve zipcode {zipcode!r} to a location and/or state; "
            "no facilities can be searched."
        )
        return {"errors": errors}

    lat, lon = latlng
    return {
        "origin_lat": lat,
        "origin_lon": lon,
        "state": resolved_state,
        "errors": errors,
    }


def _route_after_resolve(state: PipelineState) -> list[str] | str:
    if state.get("errors"):
        return "short_circuit"
    return ["fetch_snf", "fetch_alf"]


def short_circuit(state: PipelineState) -> dict[str, Any]:
    """Reached only when `resolve_location` failed. Produces an empty,
    well-formed result set rather than letting the graph crash."""
    return {
        "snf_raw": [],
        "alf_raw": [],
        "snf_enriched": [],
        "alf_enriched": [],
        "facilities": [],
        "results": [],
    }


# --- 2/3. fetch_snf / fetch_alf ------------------------------------------------


def fetch_snf(state: PipelineState) -> dict[str, Any]:
    errors = list(state.get("errors") or [])
    try:
        facilities = cms_snf.fetch_snf_facilities(state["state"])
        filtered = cms_snf.filter_by_radius(
            facilities, state["origin_lat"], state["origin_lon"], state["radius_miles"]
        )
        return {"snf_raw": filtered}
    except Exception as exc:  # noqa: BLE001 - defensive, must not crash the run
        errors.append(f"fetch_snf failed: {exc}")
        return {"snf_raw": [], "errors": errors}


def _dedupe_by_npi(facilities: list[dict]) -> list[dict]:
    """Drop exact-duplicate NPI entries.

    Empirically, NPPES's free-text-matched, `skip`-based pagination for the
    "Assisted Living" query returns the same organization (same NPI, same
    address) multiple times across pages (observed ~40% duplication rate
    for CA). `nppes_alf.fetch_alf_facilities` doesn't dedupe this itself, so
    it's handled here -- purely mechanical (first-occurrence-wins by NPI),
    not a Stage 1/2 code change, and it also cuts down on redundant Census
    geocoder calls since it runs before geocoding.
    """
    seen: set[str] = set()
    deduped = []
    for facility in facilities:
        npi = facility.get("npi")
        if npi is not None:
            if npi in seen:
                continue
            seen.add(npi)
        deduped.append(facility)
    return deduped


def fetch_alf(state: PipelineState) -> dict[str, Any]:
    errors = list(state.get("errors") or [])
    try:
        facilities = nppes_alf.fetch_alf_facilities(state["state"])
        facilities = _dedupe_by_npi(facilities)
        facilities = nppes_alf.geocode_alf_facilities(facilities)
        filtered = nppes_alf.filter_by_radius(
            facilities, state["origin_lat"], state["origin_lon"], state["radius_miles"]
        )
        return {"alf_raw": filtered}
    except Exception as exc:  # noqa: BLE001
        errors.append(f"fetch_alf failed: {exc}")
        return {"alf_raw": [], "errors": errors}


# --- 4. enrich_snf -------------------------------------------------------------


def enrich_snf(state: PipelineState) -> dict[str, Any]:
    """Attach ownership records (batch CCN lookup) and best-effort ACO
    affiliation (name-matched, see `aco_match.py`) to each SNF facility."""
    facilities = state.get("snf_raw") or []
    errors = list(state.get("errors") or [])
    if not facilities:
        return {"snf_enriched": []}

    ccns = [f["ccn"] for f in facilities if f.get("ccn")]
    try:
        ownership_by_ccn = cms_snf.fetch_ownership(ccns)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"enrich_snf: fetch_ownership failed ({exc}); ownership omitted.")
        ownership_by_ccn = {}

    try:
        matcher = aco_match.load_aco_matcher()
    except Exception as exc:  # noqa: BLE001
        errors.append(
            f"enrich_snf: ACO dataset fetch failed ({exc}); affiliated_aco left as None "
            "for all facilities (never guessed)."
        )
        matcher = None

    enriched = []
    for facility in facilities:
        facility = dict(facility)
        facility["ownership"] = ownership_by_ccn.get(facility.get("ccn"), [])
        facility["affiliated_aco"] = (
            matcher.match(facility.get("name")) if matcher is not None else None
        )
        enriched.append(facility)

    return {"snf_enriched": enriched, "errors": errors}


# --- 5. enrich_alf -------------------------------------------------------------


def enrich_alf(state: PipelineState) -> dict[str, Any]:
    """Attach best-effort bed-count enrichment from the (once-loaded)
    Stengel dataset to each ALF facility."""
    facilities = state.get("alf_raw") or []
    if not facilities:
        return {"alf_enriched": []}

    df = stengel.load_stengel_dataset()  # loaded once, reused for every facility

    enriched = []
    for facility in facilities:
        facility = dict(facility)
        facility["bed_count"] = stengel.match_bed_count(facility, df)
        enriched.append(facility)

    return {"alf_enriched": enriched}


# --- 6. reconcile (the Claude tool-calling / structured-output node) ----------


class DuplicateGroup(BaseModel):
    """A group of facility-list indices believed to refer to the same
    physical facility (e.g. a continuing-care community that shows up once
    in the SNF data and once in the ALF data)."""

    indices: list[int] = Field(
        description="0-based indices into the facility list that refer to the same physical facility (>= 2 entries)."
    )
    keep_index: int = Field(
        description="Which of `indices` to keep as the canonical/merged entry; the rest are dropped."
    )


class CleanedFacility(BaseModel):
    """A lightly-cleaned display string for one facility. Only fields that
    actually needed a cleanup should be set; leave others as `None`."""

    index: int = Field(description="0-based index into the facility list.")
    leadership: str | None = Field(
        default=None, description="Cleaned-up leadership/owner display string, or omit if unchanged."
    )
    ownership_type: str | None = Field(
        default=None, description="Cleaned-up ownership type string, or omit if unchanged."
    )
    chain_name: str | None = Field(
        default=None, description="Cleaned-up chain name string, or omit if unchanged."
    )


class ReconcileResult(BaseModel):
    duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)
    cleaned: list[CleanedFacility] = Field(default_factory=list)


def _facility_summaries(facilities: list[Facility]) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "name": f.name,
            "facility_type": f.facility_type,
            "address": f.address,
            "city": f.city,
            "state": f.state,
            "zip": f.zip,
            "leadership": f.leadership,
            "ownership_type": f.ownership_type,
            "chain_name": f.chain_name,
        }
        for i, f in enumerate(facilities)
    ]


_RECONCILE_PROMPT_TEMPLATE = """\
You are reconciling a list of elder-care facilities compiled from two \
separate government data sources: CMS (skilled nursing facilities) and \
NPPES (assisted living facilities). Some physical facilities -- especially \
continuing-care retirement communities that operate both a nursing wing and \
an assisted-living wing -- may appear once from each source under a similar \
name and address.

Given the JSON list of facilities below, do two things:

1. Identify any groups of 2+ indices that refer to the SAME physical \
facility (matching or very similar name AND a close/matching address). Only \
flag genuinely confident matches -- do not group facilities just because \
they're in the same city or have a generic name in common. For each group, \
pick which index to keep as the canonical entry (prefer the one with more \
complete address/leadership information).
2. Lightly clean up the `leadership`, `ownership_type`, and `chain_name` \
display strings for consistency and readability (fix casing, spacing, \
stray punctuation). Do NOT invent, guess, or add any new facts -- only \
reformat what's already there. Only include an entry in `cleaned` if you \
are actually changing something; omit fields within it that are unchanged.

Facilities:
{facilities_json}
"""


def _reconcile_with_llm(facilities: list[Facility]) -> list[Facility]:
    if not facilities:
        return facilities

    llm = ChatAnthropic(model=config.CLAUDE_MODEL)
    structured_llm = llm.with_structured_output(ReconcileResult)

    prompt = _RECONCILE_PROMPT_TEMPLATE.format(
        facilities_json=json.dumps(_facility_summaries(facilities), indent=2)
    )
    result = structured_llm.invoke(prompt)
    if not isinstance(result, ReconcileResult):
        # with_structured_output can return a dict depending on backend/version;
        # normalize defensively rather than assuming the type.
        result = ReconcileResult.model_validate(result)

    return _apply_reconcile_result(facilities, result)


def _apply_reconcile_result(
    facilities: list[Facility], result: ReconcileResult
) -> list[Facility]:
    facilities = list(facilities)

    # Apply string cleanup. Deliberately restricted to leadership/
    # ownership_type/chain_name -- never touches cms_overall_rating or
    # affiliated_aco, so the honesty guardrail holds even if the LLM
    # response tried to set those (the schema doesn't even expose them here).
    for cleaned in result.cleaned:
        if not (0 <= cleaned.index < len(facilities)):
            continue
        update: dict[str, Any] = {}
        if cleaned.leadership is not None:
            update["leadership"] = cleaned.leadership
        if cleaned.ownership_type is not None:
            update["ownership_type"] = cleaned.ownership_type
        if cleaned.chain_name is not None:
            update["chain_name"] = cleaned.chain_name
        if update:
            facilities[cleaned.index] = facilities[cleaned.index].model_copy(
                update=update
            )

    # Apply dedup: within each confident duplicate group, keep only keep_index.
    drop: set[int] = set()
    for group in result.duplicate_groups:
        valid = [i for i in group.indices if 0 <= i < len(facilities)]
        if len(valid) < 2:
            continue
        keep = group.keep_index if group.keep_index in valid else valid[0]
        drop.update(i for i in valid if i != keep)

    return [f for i, f in enumerate(facilities) if i not in drop]


def reconcile(state: PipelineState) -> dict[str, Any]:
    snf_enriched = state.get("snf_enriched") or []
    alf_enriched = state.get("alf_enriched") or []
    errors = list(state.get("errors") or [])

    facilities = [Facility.from_snf_dict(f) for f in snf_enriched] + [
        Facility.from_alf_dict(f) for f in alf_enriched
    ]

    if not config.ANTHROPIC_API_KEY:
        errors.append(
            "reconcile: ANTHROPIC_API_KEY not set; skipping LLM dedup/cleanup and "
            "using a plain concatenation of SNF + ALF results."
        )
        return {"facilities": facilities, "errors": errors}

    try:
        facilities = _reconcile_with_llm(facilities)
    except Exception as exc:  # noqa: BLE001 - must never crash the pipeline
        errors.append(
            f"reconcile: LLM reconciliation failed ({exc}); falling back to a plain "
            "concatenation of SNF + ALF results (no dedup/cleanup)."
        )

    return {"facilities": facilities, "errors": errors}


# --- 7. rank -------------------------------------------------------------------


def _rank_key(facility: Facility) -> tuple[float, float]:
    rating = facility.cms_overall_rating
    # None sinks below any real 1-5 rating: real ratings produce -5..-1,
    # so anything > -1 (e.g. a fixed sentinel) sorts after all of them.
    rating_key = float(-rating) if rating is not None else 1.0
    return (rating_key, facility.distance_mi)


def rank(state: PipelineState) -> dict[str, Any]:
    facilities = state.get("facilities") or []
    ranked = sorted(facilities, key=_rank_key)
    return {"results": ranked}


# --- 8. format -----------------------------------------------------------------


def format_results(state: PipelineState) -> dict[str, Any]:
    """Final cleanup pass. `rank` already produces `Facility` objects in the
    right order, so this is currently a passthrough -- kept as a separate
    node to match the pipeline design and as a seam for later formatting
    needs (e.g. the FastAPI stage) without touching `rank`'s sort logic."""
    return {"results": state.get("results") or []}


# --- graph assembly ------------------------------------------------------------


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("resolve_location", resolve_location)
    graph.add_node("fetch_snf", fetch_snf)
    graph.add_node("fetch_alf", fetch_alf)
    graph.add_node("enrich_snf", enrich_snf)
    graph.add_node("enrich_alf", enrich_alf)
    graph.add_node("reconcile", reconcile)
    graph.add_node("rank", rank)
    graph.add_node("format", format_results)
    graph.add_node("short_circuit", short_circuit)

    graph.add_edge(START, "resolve_location")
    graph.add_conditional_edges(
        "resolve_location",
        _route_after_resolve,
        ["fetch_snf", "fetch_alf", "short_circuit"],
    )

    # Independent parallel branches, fanning back in at `reconcile`.
    graph.add_edge("fetch_snf", "enrich_snf")
    graph.add_edge("fetch_alf", "enrich_alf")
    graph.add_edge(["enrich_snf", "enrich_alf"], "reconcile")

    graph.add_edge("reconcile", "rank")
    graph.add_edge("rank", "format")
    graph.add_edge("format", END)
    graph.add_edge("short_circuit", END)

    return graph.compile()


COMPILED_GRAPH = build_graph()
