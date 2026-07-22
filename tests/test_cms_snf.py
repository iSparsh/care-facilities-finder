"""Tests for care_facilities.sources.cms_snf.

Most tests mock httpx so they're deterministic and don't hit the network.
The one exception is `test_live_fetch_snf_facilities_ca`, which makes a real
call against the live CMS API; it's resilient to network issues (skips
rather than fails) and is clearly named with `_live_`.
"""

from __future__ import annotations

import httpx
import pytest

from care_facilities import cache as cache_module
from care_facilities.sources import cms_snf


# --- helpers ------------------------------------------------------------


def _provider_row(**overrides) -> dict:
    row = {
        "cms_certification_number_ccn": "055555",
        "provider_name": "Test Nursing Home",
        "provider_address": "123 Main St",
        "citytown": "Springfield",
        "state": "CA",
        "zip_code": "90001",
        "telephone_number": "5551234567",
        "ownership_type": "For profit - Corporation",
        "number_of_certified_beds": "120",
        "chain_name": "Test Chain",
        "overall_rating": "3",
        "health_inspection_rating": "2",
        "staffing_rating": "4",
        "qm_rating": "",
        "latitude": "34.05",
        "longitude": "-118.25",
    }
    row.update(overrides)
    return row


def _make_transport(pages_by_offset: dict[int, list[dict]], total_count: int):
    """Build a MockTransport that serves paginated DKAN-style responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        offset = int(params.get("offset", 0))
        page = pages_by_offset.get(offset, [])
        return httpx.Response(
            200,
            json={
                "results": page,
                "count": total_count,
                "schema": {},
                "query": {},
            },
        )

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the shared disk cache at a throwaway location for each test."""
    monkeypatch.setattr(cache_module.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cache_module, "_connection", None)
    yield
    monkeypatch.setattr(cache_module, "_connection", None)


def _patch_httpx_get(monkeypatch, transport: httpx.MockTransport):
    def fake_get(url, params=None, timeout=None):
        with httpx.Client(transport=transport) as client:
            return client.get(url, params=params, timeout=timeout)

    monkeypatch.setattr(cms_snf.httpx, "get", fake_get)


# --- fetch_snf_facilities / normalization -------------------------------


def test_fetch_snf_facilities_normalizes_ratings(monkeypatch):
    rows = [
        _provider_row(cms_certification_number_ccn="111111", overall_rating="5"),
        _provider_row(
            cms_certification_number_ccn="222222",
            overall_rating="",
            health_inspection_rating="Not Available",
            staffing_rating="9",  # out of 1-5 range -> None
        ),
    ]
    transport = _make_transport({0: rows}, total_count=2)
    _patch_httpx_get(monkeypatch, transport)

    facilities = cms_snf.fetch_snf_facilities("CA")

    assert len(facilities) == 2
    first, second = facilities

    assert first["ccn"] == "111111"
    assert first["overall_rating"] == 5
    assert first["certified_beds"] == 120
    assert first["latitude"] == 34.05
    assert first["longitude"] == -118.25

    assert second["overall_rating"] is None
    assert second["health_inspection_rating"] is None
    assert second["staffing_rating"] is None
    assert second["qm_rating"] is None


def test_fetch_snf_facilities_paginates(monkeypatch):
    # _PAGE_LIMIT is 500; simulate exactly a full first page + partial second
    # page by monkeypatching the page limit down to keep the test fast.
    monkeypatch.setattr(cms_snf, "_PAGE_LIMIT", 2)

    page0 = [
        _provider_row(cms_certification_number_ccn="111111"),
        _provider_row(cms_certification_number_ccn="222222"),
    ]
    page1 = [_provider_row(cms_certification_number_ccn="333333")]
    transport = _make_transport({0: page0, 2: page1}, total_count=3)
    _patch_httpx_get(monkeypatch, transport)

    facilities = cms_snf.fetch_snf_facilities("CA")

    assert [f["ccn"] for f in facilities] == ["111111", "222222", "333333"]


def test_fetch_snf_facilities_uses_cache(monkeypatch):
    rows = [_provider_row(cms_certification_number_ccn="111111")]
    call_count = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        call_count["n"] += 1
        transport = _make_transport({0: rows}, total_count=1)
        with httpx.Client(transport=transport) as client:
            return client.get(url, params=params, timeout=timeout)

    monkeypatch.setattr(cms_snf.httpx, "get", fake_get)

    first = cms_snf.fetch_snf_facilities("CA")
    second = cms_snf.fetch_snf_facilities("CA")

    assert first == second
    assert call_count["n"] == 1  # second call served entirely from cache


# --- filter_by_radius -----------------------------------------------------


def test_filter_by_radius_boundary_inside_and_outside():
    origin_lat, origin_lon = 0.0, 0.0

    # ~1 degree of longitude at the equator is ~69 miles.
    just_inside = {
        "ccn": "111111",
        "latitude": 0.0,
        "longitude": 0.9,  # ~62 miles
    }
    just_outside = {
        "ccn": "222222",
        "latitude": 0.0,
        "longitude": 1.5,  # ~103 miles
    }
    missing_coords = {
        "ccn": "333333",
        "latitude": None,
        "longitude": None,
    }

    result = cms_snf.filter_by_radius(
        [just_outside, just_inside, missing_coords],
        origin_lat,
        origin_lon,
        radius_miles=75,
    )

    assert [f["ccn"] for f in result] == ["111111"]
    assert result[0]["distance_mi"] > 0
    assert isinstance(result[0]["distance_mi"], float)


def test_filter_by_radius_sorts_ascending_by_distance():
    origin_lat, origin_lon = 0.0, 0.0
    far = {"ccn": "far", "latitude": 0.0, "longitude": 0.5}
    near = {"ccn": "near", "latitude": 0.0, "longitude": 0.1}

    result = cms_snf.filter_by_radius([far, near], origin_lat, origin_lon, radius_miles=100)

    assert [f["ccn"] for f in result] == ["near", "far"]
    assert result[0]["distance_mi"] <= result[1]["distance_mi"]


# --- fetch_ownership -------------------------------------------------------


def _ownership_row(**overrides) -> dict:
    row = {
        "cms_certification_number_ccn": "111111",
        "owner_name": "Some Owner LLC",
        "owner_type": "Organization",
        "role_played_by_owner_or_manager_in_facility": "OPERATIONAL/MANAGERIAL CONTROL",
        "ownership_percentage": "NOT APPLICABLE",
    }
    row.update(overrides)
    return row


def test_fetch_ownership_groups_by_ccn(monkeypatch):
    rows = [
        _ownership_row(cms_certification_number_ccn="111111", owner_name="Owner A"),
        _ownership_row(cms_certification_number_ccn="111111", owner_name="Owner B"),
        _ownership_row(
            cms_certification_number_ccn="222222",
            owner_name="Owner C",
            ownership_percentage="55.5%",
        ),
    ]
    transport = _make_transport({0: rows}, total_count=3)
    _patch_httpx_get(monkeypatch, transport)

    result = cms_snf.fetch_ownership(["111111", "222222"])

    assert set(result.keys()) == {"111111", "222222"}
    assert len(result["111111"]) == 2
    owner_names = {o["owner_name"] for o in result["111111"]}
    assert owner_names == {"Owner A", "Owner B"}
    assert result["111111"][0]["ownership_percentage"] is None

    assert result["222222"][0]["ownership_percentage"] == 55.5


def test_fetch_ownership_empty_input_returns_empty_dict():
    assert cms_snf.fetch_ownership([]) == {}


# --- fetch_aco_affiliation -------------------------------------------------


def test_fetch_aco_affiliation_returns_empty_dict_fallback():
    # Documented limitation: no CCN/NPI join key exists in the real CMS ACO
    # SNF affiliates dataset, so this is expected to always return {}.
    assert cms_snf.fetch_aco_affiliation(["111111", "222222"]) == {}


# --- live smoke test --------------------------------------------------------


def test_live_fetch_snf_facilities_ca():
    try:
        facilities = cms_snf.fetch_snf_facilities("CA")
    except httpx.HTTPError as exc:
        pytest.skip(f"network unavailable or CMS API unreachable: {exc}")

    if not facilities:
        pytest.skip("live CMS API returned no facilities for CA")

    assert len(facilities) > 1
    sample = facilities[0]
    assert sample["ccn"]
    assert sample["overall_rating"] is None or 1 <= sample["overall_rating"] <= 5
