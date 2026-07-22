"""Tests for care_facilities.geocode.

These tests use the real pgeocode dataset (no network access -- pgeocode
ships/caches a static data file), so they're deterministic and fast. We do
NOT test `geocode_address` here since it hits the live US Census Geocoder
API over the network; that's expected to be covered by later
integration/live smoke tests instead.
"""

from __future__ import annotations

import math

from care_facilities.geocode import haversine_miles, zip_to_latlng, zip_to_state


def test_zip_to_latlng_known_zip():
    result = zip_to_latlng("94404")
    assert result is not None

    lat, lng = result
    assert math.isclose(lat, 37.5, abs_tol=1.0)
    assert math.isclose(lng, -122.3, abs_tol=1.0)


def test_zip_to_state_known_zip():
    assert zip_to_state("94404") == "CA"


def test_zip_to_latlng_invalid_zip_returns_none():
    assert zip_to_latlng("00000") is None


def test_zip_to_state_invalid_zip_returns_none():
    assert zip_to_state("00000") is None


def test_haversine_miles_one_degree_longitude_at_equator():
    # One degree of longitude at the equator is ~69 miles.
    distance = haversine_miles(0.0, 0.0, 0.0, 1.0)
    assert math.isclose(distance, 69.0, abs_tol=2.0)


def test_haversine_miles_zero_distance():
    assert haversine_miles(37.5, -122.3, 37.5, -122.3) == 0.0
