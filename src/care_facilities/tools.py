"""LangChain tool wrappers around the Stage 1/2 data-source functions.

These exist so a Claude tool-calling node *could* invoke fetch/enrich steps
agentically. In practice this pipeline's graph (`graph.py`) calls the
underlying `sources.*` functions directly and deterministically for the
fetch/filter/enrich steps -- there's no real judgment call in "fetch SNFs
for state X within N miles", so agentic tool-choice would just add latency
and cost with no quality benefit. The one node that *does* use an LLM
(`reconcile`) uses structured output instead of these tools, since its job
(dedup judgment + string cleanup) is a single reasoning step over an
already-fetched list, not a multi-step tool-use loop.

These wrappers are kept around because the task calls for them explicitly,
and because they're a reasonable extension point if a future stage wants a
genuinely agentic "let Claude decide what to fetch" mode (e.g. a chat
interface that lets a user broaden/narrow a search conversationally).
"""

from __future__ import annotations

from langchain_core.tools import tool

from .sources import cms_snf, nppes_alf


@tool
def fetch_snf_facilities_in_radius(
    state: str, origin_lat: float, origin_lon: float, radius_miles: float
) -> list[dict]:
    """Fetch CMS-certified skilled nursing facilities (SNFs / nursing homes)
    registered in a US state, then filter to those within `radius_miles` of
    the given (origin_lat, origin_lon) point.

    Args:
        state: 2-letter USPS state code, e.g. "CA".
        origin_lat: Latitude of the search origin.
        origin_lon: Longitude of the search origin.
        radius_miles: Maximum distance from the origin, in miles.

    Returns a list of facility dicts (name, address, ratings, certified
    beds, ownership_type, chain_name, phone, distance_mi, ...). Does NOT
    include ownership records or ACO affiliation -- those require a
    separate enrichment step over the CCNs of the filtered results.
    """
    facilities = cms_snf.fetch_snf_facilities(state)
    return cms_snf.filter_by_radius(facilities, origin_lat, origin_lon, radius_miles)


@tool
def fetch_alf_facilities_in_radius(
    state: str, origin_lat: float, origin_lon: float, radius_miles: float
) -> list[dict]:
    """Fetch NPPES-registered assisted living facilities (ALFs) in a US
    state, geocode them (Census address geocoder, falling back to ZIP
    centroid), then filter to those within `radius_miles` of the given
    (origin_lat, origin_lon) point.

    Args:
        state: 2-letter USPS state code, e.g. "CA".
        origin_lat: Latitude of the search origin.
        origin_lon: Longitude of the search origin.
        radius_miles: Maximum distance from the origin, in miles.

    Returns a list of facility dicts (name, address, leadership_name/title,
    phone, latitude/longitude, geocode_precision, distance_mi, ...). Does
    NOT include bed-count enrichment from the Stengel dataset -- that
    requires a separate enrichment step.
    """
    facilities = nppes_alf.fetch_alf_facilities(state)
    facilities = nppes_alf.geocode_alf_facilities(facilities)
    return nppes_alf.filter_by_radius(facilities, origin_lat, origin_lon, radius_miles)
