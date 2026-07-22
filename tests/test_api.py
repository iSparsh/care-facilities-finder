"""Tests for the Stage 4 FastAPI backend (`api/main.py`).

`pipeline.run` is monkeypatched (in `api.main`'s namespace, where it's
imported) so these tests are fast, deterministic, and never touch the
network or an LLM.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from care_facilities.schema import Facility


def _snf_facility(**overrides):
    defaults = dict(
        name="Sunny Acres Nursing Home",
        facility_type="SNF",
        address="1 Test Way",
        city="Testville",
        state="CA",
        zip="90001",
        distance_mi=2.5,
        cms_overall_rating=4,
        health_inspection_rating=4,
        staffing_rating=3,
        qm_rating=4,
        certified_beds=100,
        ownership_type="For profit - Corporation",
        chain_name="Test Chain",
        affiliated_aco="Test ACO",
        leadership="Jane Doe (Administrator)",
        phone="5551234567",
        data_source="CMS",
        geocode_precision="exact",
    )
    defaults.update(overrides)
    return Facility(**defaults)


def _alf_facility(**overrides):
    defaults = dict(
        name="Golden Years Assisted Living",
        facility_type="Assisted Living",
        address="2 Test Ave",
        city="Testville",
        state="CA",
        zip="90001",
        distance_mi=5.1,
        cms_overall_rating=None,
        health_inspection_rating=None,
        staffing_rating=None,
        qm_rating=None,
        certified_beds=40,
        ownership_type=None,
        chain_name=None,
        affiliated_aco=None,
        leadership=None,
        phone="5559876543",
        data_source="NPPES",
        geocode_precision="exact",
    )
    defaults.update(overrides)
    return Facility(**defaults)


@pytest.fixture
def client():
    return TestClient(api_main.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_valid_zipcode_returns_expected_shape(client, monkeypatch):
    facilities = [_snf_facility(), _alf_facility()]

    def _fake_run(zipcode, radius_miles=None):
        assert zipcode == "94404"
        return facilities

    monkeypatch.setattr(api_main.pipeline, "run", _fake_run)

    resp = client.post("/search", json={"zipcode": "94404", "radius_miles": 15})
    assert resp.status_code == 200
    body = resp.json()

    assert body["count"] == 2
    assert body["zipcode"] == "94404"
    assert body["radius_miles"] == 15
    assert body["errors"] == []
    assert len(body["results"]) == 2

    snf_row = body["results"][0]
    assert snf_row["name"] == "Sunny Acres Nursing Home"
    assert snf_row["cms_overall_rating"] == 4
    assert snf_row["affiliated_aco"] == "Test ACO"

    alf_row = body["results"][1]
    assert alf_row["facility_type"] == "Assisted Living"
    # None fields must serialize as JSON null, not be omitted.
    assert "cms_overall_rating" in alf_row
    assert alf_row["cms_overall_rating"] is None
    assert "affiliated_aco" in alf_row
    assert alf_row["affiliated_aco"] is None
    assert "ownership_type" in alf_row
    assert alf_row["ownership_type"] is None


def test_search_uses_default_radius_when_omitted(client, monkeypatch):
    captured = {}

    def _fake_run(zipcode, radius_miles=None):
        captured["zipcode"] = zipcode
        captured["radius_miles"] = radius_miles
        return []

    monkeypatch.setattr(api_main.pipeline, "run", _fake_run)

    resp = client.post("/search", json={"zipcode": "94404"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["results"] == []
    assert captured["radius_miles"] == captured["radius_miles"]  # sanity
    assert captured["zipcode"] == "94404"


@pytest.mark.parametrize("bad_zip", ["abc", "123", "123456", "", "9440a"])
def test_search_invalid_zipcode_returns_422(client, bad_zip):
    resp = client.post("/search", json={"zipcode": bad_zip})
    assert resp.status_code == 422


def test_search_pipeline_exception_returns_clean_json_error(client, monkeypatch):
    def _raise(zipcode, radius_miles=None):
        raise RuntimeError("boom: network unreachable")

    monkeypatch.setattr(api_main.pipeline, "run", _raise)

    resp = client.post("/search", json={"zipcode": "94404"})
    # Never a raw 500 stack trace: either a clean 200 w/ errors, or a clean
    # JSON error body. Assert whichever this implementation chose is clean.
    assert resp.status_code in (200, 500)
    body = resp.json()
    assert isinstance(body, dict)
    if resp.status_code == 200:
        assert body["results"] == []
        assert body["errors"]
        assert "boom" in body["errors"][0]
    else:
        # FastAPI's default error body has a "detail" key; make sure it's
        # not a raw traceback dump either way.
        assert "detail" in body or "errors" in body


def test_root_serves_static_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Care Facilities Finder" in resp.text
    assert 'id="filter-bar"' in resp.text
    assert "Why is there no CMS rating?" in resp.text
