"""NPPES NPI Registry data source for Assisted Living Facilities (ALFs).

The NPPES NPI Registry (https://npiregistry.cms.hhs.gov/) doesn't have a
dedicated "assisted living" dataset, but organizations that operate ALFs
often register for an NPI (e.g. to bill Medicaid waiver programs, or for
home-health-adjacent services) with a taxonomy under the "Assisted Living
Facility" family (taxonomy codes starting with ``3104``). This module
queries that registry, filters down to genuine ALF taxonomy codes, and
normalizes the result into a common facility dict shape.

Important, empirically-confirmed quirks of the live API (see the live call
made while building this module):

* ``result_count`` in a page of results is the count *of that page*
  (i.e. ``len(results)``), not a running/grand total. Pagination must stop
  when a page returns fewer than the requested ``limit``.
* The ``state`` query parameter is a loose filter -- it matches organizations
  whose *any* address (mailing or location) is associated with that state
  behavior-wise in practice, but in testing we observed results whose
  ``LOCATION`` address was in a completely different state than the
  requested ``state`` (e.g. querying ``state=CA`` returned results with
  LOCATION addresses in TX, FL, PA, etc.). Because callers care about where
  the facility actually *is*, this module additionally filters client-side
  to keep only results whose ``LOCATION`` address state matches the
  requested state.
* ``taxonomy_description=Assisted+Living`` does free-text matching and pulls
  in unrelated taxonomies whose description happens to contain "assisted"
  or "living" as substrings of other words/phrases. We filter client-side to
  keep only rows with at least one taxonomy code starting with ``3104``
  (covers ``310400000X``, ``3104A0625X``, ``3104A0630X``, and any other
  future codes in that family).
"""

from __future__ import annotations

import warnings
from typing import Any

import httpx

from .. import cache, config, geocode

MAX_PAGES = 10
PAGE_LIMIT = 200

_ALF_TAXONOMY_PREFIX = "3104"


def fetch_alf_facilities(state: str) -> list[dict]:
    """Fetch assisted-living facilities registered in NPPES for `state`.

    `state` is a 2-letter US state code (e.g. "CA"). Results are filtered to
    organizations (NPI-2) with a genuine assisted-living taxonomy code, and
    whose LOCATION address is actually in `state`.

    The full fetch + parse is wrapped in `cached_call` so repeat calls within
    the TTL window don't re-hit the NPPES API. Geocoding is intentionally
    NOT performed here -- see `geocode_alf_facilities`.
    """
    state = state.strip().upper()
    cache_key = f"nppes_alf_facilities:{state}"

    def _fetch() -> list[dict]:
        raw_results = _fetch_all_pages(state)
        return _parse_results(raw_results, state)

    return cache.cached_call(cache_key, config.CACHE_TTL_NPPES, _fetch)


def _fetch_all_pages(state: str) -> list[dict]:
    """Page through the NPPES API for `state`, returning raw result dicts."""
    all_results: list[dict] = []
    skip = 0

    for page in range(MAX_PAGES):
        response = httpx.get(
            config.NPPES_API_BASE,
            params={
                "version": config.NPPES_API_VERSION,
                "enumeration_type": "NPI-2",
                "taxonomy_description": "Assisted Living",
                "state": state,
                "limit": PAGE_LIMIT,
                "skip": skip,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()

        page_results = payload.get("results", [])
        all_results.extend(page_results)

        if len(page_results) < PAGE_LIMIT:
            # Last page (NPPES's "result_count" reflects this page's size,
            # not a grand total, so this is the correct stop condition).
            break

        skip += PAGE_LIMIT
    else:
        warnings.warn(
            f"nppes_alf: hit MAX_PAGES={MAX_PAGES} page cap fetching state="
            f"{state!r} ({len(all_results)} results so far); some facilities "
            "may be missing. Increase MAX_PAGES if this is a problem."
        )

    return all_results


def _parse_results(raw_results: list[dict], state: str) -> list[dict]:
    facilities: list[dict] = []
    for result in raw_results:
        facility = _parse_one(result, state)
        if facility is not None:
            facilities.append(facility)
    return facilities


def _parse_one(result: dict[str, Any], state: str) -> dict | None:
    taxonomies = result.get("taxonomies") or []
    if not any(
        (t.get("code") or "").startswith(_ALF_TAXONOMY_PREFIX) for t in taxonomies
    ):
        return None

    basic = result.get("basic") or {}
    org_name = (basic.get("organization_name") or "").strip()
    if not org_name:
        return None

    location = _find_location_address(result.get("addresses") or [])
    if location is None:
        return None

    # Location must actually be in the requested state -- the NPPES `state`
    # query param is a loose filter (see module docstring).
    location_state = (location.get("state") or "").strip().upper()
    if location_state != state:
        return None

    address_1 = (location.get("address_1") or "").strip()
    address_2 = (location.get("address_2") or "").strip()
    if not address_1:
        return None
    address = f"{address_1}, {address_2}" if address_2 else address_1

    city = (location.get("city") or "").strip()
    zip5 = _normalize_zip(location.get("postal_code"))
    phone = (location.get("telephone_number") or "").strip() or None

    first = (basic.get("authorized_official_first_name") or "").strip()
    last = (basic.get("authorized_official_last_name") or "").strip()
    leadership_name = f"{first} {last}".strip() or None
    leadership_title = (basic.get("authorized_official_title_or_position") or "").strip() or None

    return {
        "npi": result.get("number"),
        "name": org_name,
        "address": address,
        "city": city,
        "state": location_state,
        "zip": zip5,
        "phone": phone,
        "leadership_name": leadership_name,
        "leadership_title": leadership_title,
        "latitude": None,
        "longitude": None,
    }


def _find_location_address(addresses: list[dict]) -> dict | None:
    for addr in addresses:
        if addr.get("address_purpose") == "LOCATION":
            return addr
    return None


def _normalize_zip(postal_code: str | None) -> str:
    """Normalize a possibly 9-digit (no dash) NPPES postal_code to 5 digits."""
    if not postal_code:
        return ""
    digits = "".join(ch for ch in str(postal_code) if ch.isdigit())
    return digits[:5]


def geocode_alf_facilities(facilities: list[dict]) -> list[dict]:
    """Fill in latitude/longitude for each facility via the Census geocoder.

    Falls back to a ZIP-centroid lookup (`geocode.zip_to_latlng`) when the
    Census address geocoder finds no match, marking `geocode_precision` as
    "zip_centroid" in that case (vs "address" on success).

    Intentionally not wrapped in `cached_call` at the batch level --
    `geocode_address` / `zip_to_latlng` already cache per-address/per-zip,
    so batching here would only prevent incremental reuse.
    """
    for facility in facilities:
        coords = geocode.geocode_address(
            street=facility["address"],
            city=facility["city"],
            state=facility["state"],
            zipcode=facility["zip"],
        )
        if coords is not None:
            facility["latitude"], facility["longitude"] = coords
            facility["geocode_precision"] = "address"
            continue

        fallback = geocode.zip_to_latlng(facility["zip"])
        if fallback is not None:
            facility["latitude"], facility["longitude"] = fallback
            facility["geocode_precision"] = "zip_centroid"
        else:
            facility["latitude"] = None
            facility["longitude"] = None
            facility["geocode_precision"] = None

    return facilities


def filter_by_radius(
    facilities: list[dict], origin_lat: float, origin_lon: float, radius_miles: float
) -> list[dict]:
    """Keep only facilities within `radius_miles` of (origin_lat, origin_lon).

    Adds a `distance_mi` key (rounded to 1 decimal) to each kept facility.
    Facilities with no lat/lon are excluded. Results are sorted ascending by
    distance.
    """
    in_range: list[dict] = []
    for facility in facilities:
        lat = facility.get("latitude")
        lon = facility.get("longitude")
        if lat is None or lon is None:
            continue

        distance = geocode.haversine_miles(origin_lat, origin_lon, lat, lon)
        if distance <= radius_miles:
            facility = dict(facility)
            facility["distance_mi"] = round(distance, 1)
            in_range.append(facility)

    in_range.sort(key=lambda f: f["distance_mi"])
    return in_range
