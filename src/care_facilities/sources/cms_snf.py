"""CMS-certified nursing home / skilled nursing facility (SNF) data source.

Fetches facility, ownership, and (attempted) ACO affiliation data for
CMS-certified nursing homes, using CMS's free, no-key-required data APIs:

- Provider info + star ratings: the DKAN-style "Provider Data Catalog"
  datastore query API (``{CMS_API_BASE}/{dataset_id}/0``), same family of
  API used for the ownership dataset.
- ACO SNF affiliates: a different, newer CMS "data-api/v1" API (see
  `fetch_aco_affiliation` docstring for details and why it can't currently
  be joined to CCN-keyed facility data).

Real live responses from both APIs were inspected during development (see
module docstrings below) to confirm exact field names -- CMS does not
publish a single canonical schema doc for these outside the datasets
themselves.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import httpx

from .. import cache, config
from ..geocode import haversine_miles

_REQUEST_TIMEOUT = 30.0
_PAGE_LIMIT = 500

# Values CMS uses (across various fields) to mean "no data" / "not
# applicable". Comparisons are done against the lowercased, stripped string.
_MISSING_VALUES = {
    "",
    "not available",
    "not applicable",
    "n/a",
    "na",
    "none",
    "no data available",
}

# --- CMS Provider Data Catalog (DKAN) datastore query API -------------------


def _datastore_query(dataset_id: str, conditions: list[dict[str, Any]]) -> list[dict]:
    """Page through the DKAN datastore query API for `dataset_id`, applying
    `conditions`, and return the full, flattened list of raw row dicts.

    Each condition is a dict with keys `property`, `value`, and optionally
    `operator` (defaults to "="). `value` may be a list, in which case
    `operator` should be "IN" (confirmed working against the live API).

    This performs however many HTTP requests are needed to page through all
    results (the API returns partial pages once the true "count" has been
    exhausted). Callers are expected to wrap the *entire* paginated fetch in
    `cache.cached_call` so a cache hit skips all HTTP requests, not just the
    first page.
    """
    rows: list[dict] = []
    offset = 0

    while True:
        params: dict[str, Any] = {"limit": _PAGE_LIMIT, "offset": offset}
        for i, condition in enumerate(conditions):
            params[f"conditions[{i}][property]"] = condition["property"]
            params[f"conditions[{i}][operator]"] = condition.get("operator", "=")
            value = condition["value"]
            if isinstance(value, (list, tuple, set)):
                for j, item in enumerate(value):
                    params[f"conditions[{i}][value][{j}]"] = item
            else:
                params[f"conditions[{i}][value]"] = value

        response = httpx.get(
            f"{config.CMS_API_BASE}/{dataset_id}/0",
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()

        # Live inspection (2026-07) confirmed the response shape is a dict
        # with a "results" list (plus "count", "schema", "query" keys), not
        # a bare list as the DKAN docs sometimes suggest.
        page = payload.get("results", []) if isinstance(payload, dict) else payload
        if not page:
            break

        rows.extend(page)

        if len(page) < _PAGE_LIMIT:
            # Partial (or empty) page means we've reached the end.
            break
        offset += _PAGE_LIMIT

    return rows


# --- Value parsing helpers ---------------------------------------------------


def _is_missing(raw: Any) -> bool:
    if raw is None:
        return True
    return str(raw).strip().lower() in _MISSING_VALUES


def _parse_int(raw: Any) -> int | None:
    if _is_missing(raw):
        return None
    try:
        return int(float(str(raw).strip()))
    except ValueError:
        return None


def _parse_float(raw: Any) -> float | None:
    if _is_missing(raw):
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _parse_rating(raw: Any) -> int | None:
    """Parse a CMS 1-5 star rating field. Non-numeric placeholder strings
    (blank, "Not Available", etc.) and out-of-range values become None."""
    value = _parse_int(raw)
    if value is None or not (1 <= value <= 5):
        return None
    return value


def _parse_percentage(raw: Any) -> float | None:
    """Parse the ownership dataset's `ownership_percentage` field.

    CMS frequently reports this as "NOT APPLICABLE" or blank (very common --
    that's expected, not an error), and otherwise as a numeric string
    sometimes suffixed with "%".
    """
    if _is_missing(raw):
        return None
    cleaned = str(raw).strip().rstrip("%").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _clean_str(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


# --- Facility info + ratings -------------------------------------------------


def _normalize_facility(row: dict) -> dict:
    return {
        "ccn": _clean_str(row.get("cms_certification_number_ccn")),
        "name": _clean_str(row.get("provider_name")),
        "address": _clean_str(row.get("provider_address")),
        "city": _clean_str(row.get("citytown")),
        "state": _clean_str(row.get("state")),
        "zip": _clean_str(row.get("zip_code")),
        "latitude": _parse_float(row.get("latitude")),
        "longitude": _parse_float(row.get("longitude")),
        "overall_rating": _parse_rating(row.get("overall_rating")),
        "health_inspection_rating": _parse_rating(row.get("health_inspection_rating")),
        "staffing_rating": _parse_rating(row.get("staffing_rating")),
        "qm_rating": _parse_rating(row.get("qm_rating")),
        "certified_beds": _parse_int(row.get("number_of_certified_beds")),
        "ownership_type": _clean_str(row.get("ownership_type")),
        "chain_name": _clean_str(row.get("chain_name")),
        "phone": _clean_str(row.get("telephone_number")),
    }


def fetch_snf_facilities(state: str) -> list[dict]:
    """Fetch all CMS-certified SNF/nursing home facilities in `state`.

    `state` is a 2-letter USPS state code (e.g. "CA"). Results are cached
    (fully paginated + parsed) under `f"cms_snf_facilities:{state}"` for
    `config.CACHE_TTL_CMS` seconds, so a cache hit performs zero HTTP calls.

    Real field names confirmed via a live query against dataset
    `4pq5-n9py` (2026-07): cms_certification_number_ccn, provider_name,
    provider_address, citytown, state, zip_code, telephone_number,
    ownership_type, number_of_certified_beds, chain_name, overall_rating,
    health_inspection_rating, staffing_rating, qm_rating, latitude,
    longitude (both present as string-encoded decimals). Missing ratings
    come through as an empty string, not a placeholder word.
    """
    normalized_state = state.strip().upper()
    cache_key = f"cms_snf_facilities:{normalized_state}"

    def _fetch() -> list[dict]:
        raw_rows = _datastore_query(
            config.CMS_SNF_PROVIDER_INFO_DATASET,
            conditions=[
                {"property": "state", "value": normalized_state, "operator": "="}
            ],
        )
        return [_normalize_facility(row) for row in raw_rows]

    return cache.cached_call(cache_key, config.CACHE_TTL_CMS, _fetch)


def filter_by_radius(
    facilities: list[dict],
    origin_lat: float,
    origin_lon: float,
    radius_miles: float,
) -> list[dict]:
    """Filter `facilities` to those within `radius_miles` of (origin_lat,
    origin_lon), adding a `distance_mi` key (float, rounded to 1 decimal).

    Facilities with missing latitude/longitude are excluded defensively
    (CMS data should always have these, but we don't want a bad row to
    crash the whole pipeline). Results are sorted ascending by distance.
    """
    results: list[dict] = []

    for facility in facilities:
        lat = facility.get("latitude")
        lon = facility.get("longitude")
        if lat is None or lon is None:
            continue

        distance = haversine_miles(origin_lat, origin_lon, lat, lon)
        if distance <= radius_miles:
            enriched = dict(facility)
            enriched["distance_mi"] = round(distance, 1)
            results.append(enriched)

    results.sort(key=lambda f: f["distance_mi"])
    return results


# --- Ownership ---------------------------------------------------------------


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _ccn_batch_cache_key(ccns: list[str]) -> str:
    normalized = ",".join(sorted(set(ccns)))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"cms_snf_ownership:{digest}"


def fetch_ownership(ccns: list[str]) -> dict[str, list[dict]]:
    """Fetch SNF ownership/management records for the given CCNs.

    Queries dataset `y2hd-n93e` (one row per owner/manager per facility),
    filtering server-side with an `IN` condition on
    `cms_certification_number_ccn` (confirmed working live: the DKAN
    datastore query API accepts
    `conditions[0][operator]=IN` with `conditions[0][value][N]=...` list
    values). Requests are batched (100 CCNs per request) to keep query
    strings a reasonable size.

    Returns `{ccn: [{"owner_name", "owner_type", "role",
    "ownership_percentage"}, ...]}`. `ownership_percentage` is frequently
    `None` -- CMS marks it "NOT APPLICABLE" for the vast majority of
    owner/manager rows, which is expected, not a parsing failure.

    Cache key is a hash of the sorted, de-duplicated CCN list, so the exact
    same CCN set is served from cache; TTL is `config.CACHE_TTL_CMS`.
    """
    unique_ccns = sorted({c for c in ccns if c})
    if not unique_ccns:
        return {}

    cache_key = _ccn_batch_cache_key(unique_ccns)

    def _fetch() -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}

        for batch in _chunked(unique_ccns, 100):
            raw_rows = _datastore_query(
                config.CMS_SNF_OWNERSHIP_DATASET,
                conditions=[
                    {
                        "property": "cms_certification_number_ccn",
                        "value": batch,
                        "operator": "IN",
                    }
                ],
            )
            for row in raw_rows:
                ccn = _clean_str(row.get("cms_certification_number_ccn"))
                if not ccn:
                    continue
                grouped.setdefault(ccn, []).append(
                    {
                        "owner_name": _clean_str(row.get("owner_name")),
                        "owner_type": _clean_str(row.get("owner_type")),
                        "role": _clean_str(
                            row.get("role_played_by_owner_or_manager_in_facility")
                        ),
                        "ownership_percentage": _parse_percentage(
                            row.get("ownership_percentage")
                        ),
                    }
                )

        return grouped

    return cache.cached_call(cache_key, config.CACHE_TTL_CMS, _fetch)


# --- ACO SNF affiliation ------------------------------------------------------

# Real dataset found via CMS's data catalog (https://data.cms.gov/data.json,
# 2026-07): "Accountable Care Organization Skilled Nursing Facility
# Affiliates" (Medicare Shared Savings Program). It IS a real, queryable,
# no-key API -- confirmed live:
#
#   GET https://data.cms.gov/data-api/v1/dataset/5b227bd9-82d4-4145-86fd-809e02ca7f18/data
#       ?size=5000&offset=0
#
# This is a *different* API family than the provider-data DKAN API used
# above (CMS's newer "data-api/v1"), returning a bare JSON list rather than
# a {"results": [...]} envelope. Live response fields, confirmed 2026-07:
#   ACO_ID, Aff_LBN, ACO_Name, ACO_Service_Area, Agreement_Period_Num,
#   Initial_Start_Date, Current_Start_Date, "Re-entering_ACO", BASIC_Track,
#   BASIC_Track_Level, ENHANCED_Track, High_Revenue_ACO, Low_Revenue_ACO,
#   Adv_Pay, AIM, AIP, PSS, "SNF_3-Day_Rule_Waiver", Prospective_Assignment,
#   Retrospective_Assignment, ACO_Address, ACO_Public_Reporting_Website,
#   ACO_Exec_Name/Email/Phone, ACO_Public_Name/Email/Phone,
#   ACO_Compliance_Contact_Name, ACO_Medical_Director_Name,
#   pc_flex_agreement_status.
# (A CSV download of the same data is also published, e.g.
#  https://data.cms.gov/sites/default/files/2026-01/.../PY2026_Medicare_
#  Shared_Savings_Program_ACO_SNF_Affiliates.csv -- same fields.)
#
# THE BLOCKER: there is no CMS Certification Number (CCN) or NPI field
# anywhere in this dataset for the affiliated SNF. The only facility
# identifier is `Aff_LBN` (the affiliate's *legal business name*, free
# text), and `ACO_Address` is the ACO's own mailing address, not the
# facility's -- so there isn't even an address to cross-check against.
# `fetch_snf_facilities` does expose a `legal_business_name`-equivalent on
# the raw CMS rows in principle, but joining `Aff_LBN` to a CCN would
# require fuzzy/normalized name matching (case, punctuation, "LLC" vs
# "L.L.C.", multiple facilities under a chain sharing a legal name, no
# state/address to disambiguate) -- not a clean, reliable join, and not
# something that belongs behind a function whose only input is a CCN list.
#
# DECISION: per the task's documented fallback, this returns `{}` for now.
#
# What a human or later integration-stage agent would need to do to make
# this real:
#   1. Fetch the full ACO SNF affiliates dataset (paginate
#      `.../data-api/v1/dataset/5b227bd9-82d4-4145-86fd-809e02ca7f18/data`
#      with `size`/`offset`, or download the published CSV directly).
#   2. Fetch the full facility list (via `fetch_snf_facilities`, which
#      exposes each facility's `name`; the raw CMS row also has a distinct
#      `legal_business_name` field not currently surfaced in the normalized
#      dict -- could be added if needed for matching).
#   3. Implement + manually QA a fuzzy name-matching step (e.g. rapidfuzz,
#      normalized on case/punctuation/corporate suffixes) to associate
#      `Aff_LBN` rows with CCNs, accepting some false negatives/positives.
#   4. Periodically re-check whether CMS adds a CCN/NPI column to this
#      dataset in a future release -- would make this whole workaround
#      unnecessary.
ACO_SNF_AFFILIATES_DATASET_ID = "5b227bd9-82d4-4145-86fd-809e02ca7f18"
ACO_SNF_AFFILIATES_API_BASE = "https://data.cms.gov/data-api/v1/dataset"


def fetch_aco_affiliation(ccns: list[str]) -> dict[str, dict]:
    """Attempt to fetch ACO (Accountable Care Organization) SNF-affiliate
    data for the given CCNs.

    LIMITATION (see the module-level comment above `ACO_SNF_AFFILIATES_
    DATASET_ID` for the full writeup): the real CMS "ACO Skilled Nursing
    Facility Affiliates" dataset was located and confirmed to have a
    working, no-key API, but it has no CCN/NPI field -- only a free-text
    affiliate legal business name -- so it cannot be reliably joined to a
    CCN list without fuzzy name matching. That matching step is out of
    scope here; this function currently always returns `{}`.
    """
    return {}
