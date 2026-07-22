"""Tests for care_facilities.sources.nppes_alf.

Most tests mock httpx (and geocode.geocode_address / zip_to_latlng) so they
run deterministically without hitting the network or the on-disk cache in
a way that depends on prior runs. One test (name containing `_live_`) makes
a real call to the NPPES API and skips gracefully on network failure.
"""

from __future__ import annotations

import httpx
import pytest

from care_facilities import cache as cache_module
from care_facilities.sources import nppes_alf


# --- Fixtures / fakes -------------------------------------------------------


def _make_result(
    npi="1111111111",
    org_name="SUNNY ACRES ASSISTED LIVING LLC",
    taxonomy_code="310400000X",
    location_state="CA",
    postal_code="900034634",
    address_1="123 MAIN ST",
    address_2=None,
    first_name="JANE",
    last_name="SMITH",
    title="Administrator",
    include_location=True,
):
    addresses = [
        {
            "address_1": "PO BOX 1",
            "address_purpose": "MAILING",
            "city": "SACRAMENTO",
            "state": "CA",
            "postal_code": "958140001",
            "telephone_number": "916-555-0000",
        }
    ]
    if include_location:
        location = {
            "address_1": address_1,
            "address_purpose": "LOCATION",
            "city": "LOS ANGELES",
            "state": location_state,
            "postal_code": postal_code,
            "telephone_number": "213-555-1234",
        }
        if address_2:
            location["address_2"] = address_2
        addresses.append(location)

    return {
        "number": npi,
        "basic": {
            "organization_name": org_name,
            "authorized_official_first_name": first_name,
            "authorized_official_last_name": last_name,
            "authorized_official_title_or_position": title,
        },
        "addresses": addresses,
        "taxonomies": [{"code": taxonomy_code, "desc": "whatever", "primary": True}],
    }


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _clear_cache(tmp_path, monkeypatch):
    """Point the sqlite cache at a fresh temp dir per test so tests don't
    interfere with each other (or with a real ~/.cache from prior runs)."""
    monkeypatch.chdir(tmp_path)
    # Reset the module-level cached connection so it reopens against the
    # new (temp) cache dir.
    cache_module._connection = None
    yield
    cache_module._connection = None


# --- fetch_alf_facilities / taxonomy + zip + state filtering ----------------


def test_fetch_alf_facilities_filters_taxonomy_and_normalizes_zip(monkeypatch):
    page = {
        "result_count": 3,
        "results": [
            _make_result(npi="1", org_name="GOOD ALF ONE", taxonomy_code="310400000X"),
            _make_result(npi="2", org_name="GOOD ALF TWO", taxonomy_code="3104A0625X"),
            _make_result(
                npi="3",
                org_name="UNRELATED CLINIC",
                taxonomy_code="207Q00000X",  # family medicine, not ALF
            ),
        ],
    }

    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse(page)

    monkeypatch.setattr(httpx, "get", fake_get)

    facilities = nppes_alf.fetch_alf_facilities("CA")

    assert len(calls) == 1  # single page since len(results) < limit (200)
    names = {f["name"] for f in facilities}
    assert names == {"GOOD ALF ONE", "GOOD ALF TWO"}

    one = next(f for f in facilities if f["npi"] == "1")
    assert one["zip"] == "90003"  # normalized from 9-digit "900034634"
    assert one["state"] == "CA"
    assert one["leadership_name"] == "JANE SMITH"
    assert one["leadership_title"] == "Administrator"
    assert one["latitude"] is None
    assert one["longitude"] is None


def test_fetch_alf_facilities_drops_results_with_no_org_name_or_location(monkeypatch):
    page = {
        "result_count": 2,
        "results": [
            _make_result(npi="1", org_name=""),  # no org name -> dropped
            _make_result(npi="2", include_location=False),  # no LOCATION -> dropped
        ],
    }

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(page))

    facilities = nppes_alf.fetch_alf_facilities("CA")
    assert facilities == []


def test_fetch_alf_facilities_drops_results_whose_location_is_in_another_state(monkeypatch):
    # Regression test for a real API quirk: querying state=CA can return
    # results whose LOCATION address is actually in a different state.
    page = {
        "result_count": 1,
        "results": [_make_result(npi="1", location_state="FL")],
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResponse(page))

    facilities = nppes_alf.fetch_alf_facilities("CA")
    assert facilities == []


def test_fetch_alf_facilities_paginates_until_short_page(monkeypatch):
    full_page = {
        "result_count": nppes_alf.PAGE_LIMIT,
        "results": [
            _make_result(npi=str(i), org_name=f"ALF {i}")
            for i in range(nppes_alf.PAGE_LIMIT)
        ],
    }
    short_page = {
        "result_count": 1,
        "results": [_make_result(npi="last", org_name="LAST ALF")],
    }

    responses = [full_page, short_page]
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params["skip"])
        return _FakeResponse(responses.pop(0))

    monkeypatch.setattr(httpx, "get", fake_get)

    facilities = nppes_alf.fetch_alf_facilities("CA")

    assert calls == [0, nppes_alf.PAGE_LIMIT]
    assert len(facilities) == nppes_alf.PAGE_LIMIT + 1


def test_fetch_alf_facilities_uses_cache(monkeypatch):
    page = {
        "result_count": 1,
        "results": [_make_result(npi="1", org_name="CACHED ALF")],
    }
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return _FakeResponse(page)

    monkeypatch.setattr(httpx, "get", fake_get)

    first = nppes_alf.fetch_alf_facilities("CA")
    second = nppes_alf.fetch_alf_facilities("CA")

    assert first == second
    assert len(calls) == 1  # second call served from cache, no new HTTP call


# --- geocode_alf_facilities --------------------------------------------------


def test_geocode_alf_facilities_uses_address_geocode_when_available(monkeypatch):
    facility = {
        "npi": "1",
        "name": "ALF",
        "address": "123 MAIN ST",
        "city": "LOS ANGELES",
        "state": "CA",
        "zip": "90003",
        "latitude": None,
        "longitude": None,
    }

    monkeypatch.setattr(
        nppes_alf.geocode, "geocode_address", lambda **kwargs: (34.0, -118.0)
    )
    monkeypatch.setattr(
        nppes_alf.geocode,
        "zip_to_latlng",
        lambda *a, **k: (999.0, 999.0),  # should NOT be used
    )

    result = nppes_alf.geocode_alf_facilities([facility])

    assert result[0]["latitude"] == 34.0
    assert result[0]["longitude"] == -118.0
    assert result[0]["geocode_precision"] == "address"


def test_geocode_alf_facilities_falls_back_to_zip_centroid(monkeypatch):
    facility = {
        "npi": "1",
        "name": "ALF",
        "address": "UNMATCHABLE ADDRESS",
        "city": "LOS ANGELES",
        "state": "CA",
        "zip": "90003",
        "latitude": None,
        "longitude": None,
    }

    monkeypatch.setattr(nppes_alf.geocode, "geocode_address", lambda **kwargs: None)
    monkeypatch.setattr(
        nppes_alf.geocode, "zip_to_latlng", lambda zipcode: (34.05, -118.25)
    )

    result = nppes_alf.geocode_alf_facilities([facility])

    assert result[0]["latitude"] == 34.05
    assert result[0]["longitude"] == -118.25
    assert result[0]["geocode_precision"] == "zip_centroid"


def test_geocode_alf_facilities_handles_total_failure(monkeypatch):
    facility = {
        "npi": "1",
        "name": "ALF",
        "address": "UNMATCHABLE ADDRESS",
        "city": "NOWHERE",
        "state": "CA",
        "zip": "00000",
        "latitude": None,
        "longitude": None,
    }

    monkeypatch.setattr(nppes_alf.geocode, "geocode_address", lambda **kwargs: None)
    monkeypatch.setattr(nppes_alf.geocode, "zip_to_latlng", lambda zipcode: None)

    result = nppes_alf.geocode_alf_facilities([facility])

    assert result[0]["latitude"] is None
    assert result[0]["longitude"] is None
    assert result[0]["geocode_precision"] is None


# --- filter_by_radius --------------------------------------------------------


def test_filter_by_radius_excludes_missing_coords_and_out_of_range():
    origin_lat, origin_lon = 37.7749, -122.4194  # San Francisco

    facilities = [
        {"name": "near", "latitude": 37.7849, "longitude": -122.4094},  # ~0.9mi
        {"name": "far", "latitude": 34.0522, "longitude": -118.2437},  # LA, ~350mi
        {"name": "no_coords", "latitude": None, "longitude": None},
    ]

    result = nppes_alf.filter_by_radius(facilities, origin_lat, origin_lon, radius_miles=25)

    names = [f["name"] for f in result]
    assert names == ["near"]
    assert "distance_mi" in result[0]
    assert result[0]["distance_mi"] < 25


def test_filter_by_radius_sorts_ascending_by_distance():
    origin_lat, origin_lon = 37.7749, -122.4194

    facilities = [
        {"name": "far", "latitude": 37.6, "longitude": -122.1},
        {"name": "near", "latitude": 37.78, "longitude": -122.42},
    ]

    result = nppes_alf.filter_by_radius(facilities, origin_lat, origin_lon, radius_miles=100)

    assert [f["name"] for f in result] == ["near", "far"]
    assert result[0]["distance_mi"] <= result[1]["distance_mi"]


# --- live integration test (skips gracefully on network failure) -----------


def test_fetch_alf_facilities_live_ca():
    try:
        facilities = nppes_alf.fetch_alf_facilities("CA")
    except Exception as exc:  # network hiccup, API down, etc.
        pytest.skip(f"live NPPES call failed: {exc}")

    if not facilities:
        pytest.skip("live NPPES call returned no results (unexpected but not fatal)")

    assert len(facilities) >= 1
    first = facilities[0]
    assert first["name"]
    assert first["address"]
    assert first["state"] == "CA"
