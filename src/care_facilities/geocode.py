"""Geocoding utilities: zipcode -> lat/lng, zipcode -> state, address -> lat/lng,
and great-circle distance between two points.
"""

from __future__ import annotations

import math
import warnings

import httpx
import pgeocode

from . import cache, config

_NOMINATIM_US = pgeocode.Nominatim("us")

EARTH_RADIUS_MILES = 3958.8


def zip_to_latlng(zipcode: str) -> tuple[float, float] | None:
    """Resolve a US zipcode to (latitude, longitude).

    Returns None if the zipcode is not found in the pgeocode dataset.
    """
    record = _NOMINATIM_US.query_postal_code(zipcode)
    lat = record.latitude
    lng = record.longitude

    if _is_nan(lat) or _is_nan(lng):
        return None

    return (float(lat), float(lng))


def zip_to_state(zipcode: str) -> str | None:
    """Resolve a US zipcode to its 2-letter state code.

    Returns None if the zipcode is not found in the pgeocode dataset.
    """
    record = _NOMINATIM_US.query_postal_code(zipcode)
    state_code = record.state_code

    if state_code is None or _is_nan(state_code):
        return None

    state_code = str(state_code).strip()
    if not state_code:
        return None

    return state_code


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lng points, in miles."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_MILES * c


def geocode_address(
    street: str, city: str, state: str, zipcode: str
) -> tuple[float, float] | None:
    """Geocode a full street address via the US Census Geocoder.

    Returns (latitude, longitude), or None if there's no match or the request
    fails for any reason (network error, timeout, non-200 response, malformed
    response). Callers should fall back to a zip-centroid lookup (via
    `zip_to_latlng`) when this returns None.

    Note: this function is intentionally not covered by a live/networked unit
    test here (to avoid flaky CI); it's expected to be exercised by later
    integration / live smoke tests.
    """
    normalized = f"{street.strip()}|{city.strip()}|{state.strip()}|{zipcode.strip()}".lower()
    cache_key = f"census_geocode:{normalized}"

    def _fetch() -> dict | None:
        try:
            response = httpx.get(
                f"{config.CENSUS_GEOCODER_BASE}/locations/address",
                params={
                    "street": street,
                    "city": city,
                    "state": state,
                    "zip": zipcode,
                    "benchmark": "Public_AR_Current",
                    "format": "json",
                },
                timeout=3.0,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            warnings.warn(f"Census geocoder request failed: {exc}")
            return None

    payload = cache.cached_call(cache_key, config.CACHE_TTL_GEOCODE, _fetch)

    if not payload:
        return None

    try:
        matches = payload["result"]["addressMatches"]
        if not matches:
            return None

        coordinates = matches[0]["coordinates"]
        lng = float(coordinates["x"])
        lat = float(coordinates["y"])
        return (lat, lng)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        warnings.warn(f"Could not parse Census geocoder response: {exc}")
        return None


def _is_nan(value: object) -> bool:
    try:
        return math.isnan(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
