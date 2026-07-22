"""LangGraph pipeline: zipcode -> ranked list of `Facility` results.

Pipeline shape (see module-level `build_graph()`):

    resolve_location -> fetch_snf  \\                    /-> reconcile -> rank -> format -> END
                     -> fetch_alf  /-> enrich_alf --------/
                                   \\-> enrich_snf --------/

(fetch_snf/fetch_alf and enrich_snf/enrich_alf each run as independent
parallel branches that fan back in at `reconcile`.)

Every fetch/filter/enrich/reconcile node is a deterministic wrapper around
the Stage 1/2 `sources.*` functions or plain Python -- no LLM involved, and
each node defends itself with a try/except so a single flaky HTTP call
degrades to an empty list rather than crashing the whole run.

`reconcile` flags likely duplicate SNF/ALF entries (e.g. continuing-care
communities that show up in both source datasets) using the same
conservative fuzzy-name-matching approach as `aco_match.py` (difflib,
normalized names, high similarity cutoff), combined with a matching ZIP
code -- never an LLM call, so it's free, fast, and has no external API
dependency. It would rather under-match (leave two entries separate) than
over-match (silently merge two different facilities), consistent with this
project's honesty-guardrail philosophy elsewhere.
"""

from __future__ import annotations

import difflib
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import aco_match, geocode, progress
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
    progress.emit("resolve_location", f"Resolving zipcode {zipcode}…")

    latlng = geocode.zip_to_latlng(zipcode)
    resolved_state = geocode.zip_to_state(zipcode)

    if latlng is None or resolved_state is None:
        msg = (
            f"Could not resolve zipcode {zipcode!r} to a location and/or state; "
            "no facilities can be searched."
        )
        errors.append(msg)
        progress.emit("resolve_location", msg, level="error")
        return {"errors": errors}

    lat, lon = latlng
    progress.emit(
        "resolve_location",
        f"Located {zipcode} in {resolved_state}.",
        state=resolved_state,
    )
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
    progress.emit("fetch_snf", f"Fetching nursing homes in {state['state']}…")
    try:
        facilities = cms_snf.fetch_snf_facilities(state["state"])
        filtered = cms_snf.filter_by_radius(
            facilities, state["origin_lat"], state["origin_lon"], state["radius_miles"]
        )
        progress.emit(
            "fetch_snf",
            f"Found {len(filtered)} nursing home(s) within {state['radius_miles']} mi.",
            count=len(filtered),
        )
        return {"snf_raw": filtered}
    except Exception as exc:  # noqa: BLE001 - defensive, must not crash the run
        msg = f"fetch_snf failed: {exc}"
        errors.append(msg)
        progress.emit("fetch_snf", msg, level="error")
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
    progress.emit(
        "fetch_alf", f"Fetching assisted-living facilities in {state['state']}…"
    )
    try:
        facilities = nppes_alf.fetch_alf_facilities(state["state"])
        facilities = _dedupe_by_npi(facilities)
        progress.emit(
            "fetch_alf",
            f"Loaded {len(facilities)} assisted-living record(s); narrowing by ZIP…",
            count=len(facilities),
        )
        # Cheap ZIP-centroid prefilter so we don't Census-geocode an entire state.
        facilities = nppes_alf.prefilter_by_zip_centroid(
            facilities,
            state["origin_lat"],
            state["origin_lon"],
            state["radius_miles"],
        )
        progress.emit(
            "fetch_alf",
            f"{len(facilities)} candidate(s) near the search area; geocoding addresses…",
            count=len(facilities),
        )
        facilities = nppes_alf.geocode_alf_facilities(facilities)
        filtered = nppes_alf.filter_by_radius(
            facilities, state["origin_lat"], state["origin_lon"], state["radius_miles"]
        )
        progress.emit(
            "fetch_alf",
            f"Found {len(filtered)} assisted-living facilit(ies) within "
            f"{state['radius_miles']} mi.",
            count=len(filtered),
        )
        return {"alf_raw": filtered}
    except Exception as exc:  # noqa: BLE001
        msg = f"fetch_alf failed: {exc}"
        errors.append(msg)
        progress.emit("fetch_alf", msg, level="error")
        return {"alf_raw": [], "errors": errors}


# --- 4. enrich_snf -------------------------------------------------------------


def enrich_snf(state: PipelineState) -> dict[str, Any]:
    """Attach ownership records (batch CCN lookup) and best-effort ACO
    affiliation (name-matched, see `aco_match.py`) to each SNF facility."""
    facilities = state.get("snf_raw") or []
    errors = list(state.get("errors") or [])
    if not facilities:
        return {"snf_enriched": []}

    progress.emit("enrich_snf", "Enriching nursing homes (ownership + ACO)…")
    ccns = [f["ccn"] for f in facilities if f.get("ccn")]
    try:
        ownership_by_ccn = cms_snf.fetch_ownership(ccns)
    except Exception as exc:  # noqa: BLE001
        msg = f"enrich_snf: fetch_ownership failed ({exc}); ownership omitted."
        errors.append(msg)
        progress.emit("enrich_snf", msg, level="error")
        ownership_by_ccn = {}

    try:
        matcher = aco_match.load_aco_matcher()
    except Exception as exc:  # noqa: BLE001
        msg = (
            f"enrich_snf: ACO dataset fetch failed ({exc}); affiliated_aco left as None "
            "for all facilities (never guessed)."
        )
        errors.append(msg)
        progress.emit("enrich_snf", msg, level="error")
        matcher = None

    enriched = []
    for facility in facilities:
        facility = dict(facility)
        facility["ownership"] = ownership_by_ccn.get(facility.get("ccn"), [])
        facility["affiliated_aco"] = (
            matcher.match(facility.get("name")) if matcher is not None else None
        )
        enriched.append(facility)

    progress.emit("enrich_snf", f"Enriched {len(enriched)} nursing home(s).")
    return {"snf_enriched": enriched, "errors": errors}


# --- 5. enrich_alf -------------------------------------------------------------


def enrich_alf(state: PipelineState) -> dict[str, Any]:
    """Attach best-effort bed-count enrichment from the (once-loaded)
    Stengel dataset to each ALF facility."""
    facilities = state.get("alf_raw") or []
    if not facilities:
        return {"alf_enriched": []}

    progress.emit("enrich_alf", "Enriching assisted-living bed counts…")
    df = stengel.load_stengel_dataset()  # loaded once, reused for every facility

    enriched = []
    for facility in facilities:
        facility = dict(facility)
        facility["bed_count"] = stengel.match_bed_count(facility, df)
        enriched.append(facility)

    progress.emit("enrich_alf", f"Enriched {len(enriched)} assisted-living facilit(ies).")
    return {"alf_enriched": enriched}


# --- 6. reconcile (naive, deterministic dedup) --------------------------------

# Same spirit as aco_match.MATCH_CUTOFF: deliberately high so we'd rather
# leave two genuinely-different facilities separate than wrongly merge them.
_DEDUP_NAME_CUTOFF = 0.87


def _find_duplicate_groups(facilities: list[Facility]) -> list[list[int]]:
    """Group facility indices that look like the same physical facility.

    A pair is considered a duplicate only if BOTH: (a) normalized names are
    a close/exact fuzzy match (same cutoff style as `aco_match.py`), and
    (b) they share a ZIP code (a cheap, reliable proxy for "same location"
    given `Facility` carries no lat/lon of its own, only distance-to-search-
    origin). This catches the documented case -- continuing-care
    communities registered once in CMS (SNF) and once in NPPES (ALF) under
    a near-identical name -- without needing any network/LLM call.
    """
    normalized = [aco_match.normalize_name(f.name) for f in facilities]

    groups: list[list[int]] = []
    assigned: dict[int, int] = {}  # facility index -> group index

    for i in range(len(facilities)):
        if not normalized[i] or facilities[i].zip == "":
            continue
        for j in range(i + 1, len(facilities)):
            if not normalized[j] or facilities[i].zip != facilities[j].zip:
                continue
            close = normalized[i] == normalized[j] or difflib.SequenceMatcher(
                None, normalized[i], normalized[j]
            ).ratio() >= _DEDUP_NAME_CUTOFF
            if not close:
                continue

            gi = assigned.get(i)
            gj = assigned.get(j)
            if gi is not None:
                groups[gi].append(j)
                assigned[j] = gi
            elif gj is not None:
                groups[gj].append(i)
                assigned[i] = gj
            else:
                groups.append([i, j])
                assigned[i] = assigned[j] = len(groups) - 1

    return groups


def _completeness_score(facility: Facility) -> int:
    """More non-null/non-empty informative fields = more complete. Used to
    pick which entry in a duplicate group to keep."""
    fields = (
        facility.leadership,
        facility.ownership_type,
        facility.chain_name,
        facility.phone,
        facility.certified_beds,
        facility.cms_overall_rating,
    )
    return sum(1 for value in fields if value not in (None, ""))


def _dedupe_facilities(facilities: list[Facility]) -> list[Facility]:
    groups = _find_duplicate_groups(facilities)
    drop: set[int] = set()
    for group in groups:
        # Prefer the SNF entry (carries CMS ratings/ownership -- strictly
        # more information than the ALF side of a matched pair), then the
        # most complete entry, as a tiebreak.
        keep = max(
            group,
            key=lambda i: (
                facilities[i].facility_type == "SNF",
                _completeness_score(facilities[i]),
            ),
        )
        drop.update(i for i in group if i != keep)
    return [f for i, f in enumerate(facilities) if i not in drop]


def reconcile(state: PipelineState) -> dict[str, Any]:
    snf_enriched = state.get("snf_enriched") or []
    alf_enriched = state.get("alf_enriched") or []
    errors = list(state.get("errors") or [])

    progress.emit("reconcile", "Merging duplicate SNF/ALF entries…")
    facilities = [Facility.from_snf_dict(f) for f in snf_enriched] + [
        Facility.from_alf_dict(f) for f in alf_enriched
    ]
    before = len(facilities)
    facilities = _dedupe_facilities(facilities)
    progress.emit(
        "reconcile",
        f"Reconciled {before} → {len(facilities)} facilit(ies).",
        count=len(facilities),
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
    progress.emit("rank", "Ranking results…")
    ranked = sorted(facilities, key=_rank_key)
    progress.emit("rank", f"Ready — {len(ranked)} facilit(ies).", count=len(ranked))
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
