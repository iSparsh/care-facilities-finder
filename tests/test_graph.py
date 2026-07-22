"""Unit tests for the LangGraph pipeline nodes in care_facilities.graph.

These are all deterministic / no-network / no-LLM tests:
- `rank` sort order (None ratings sink below rated facilities; distance
  tiebreaks).
- The ACO name-matching helper (care_facilities.aco_match).
- The `reconcile` node's naive, deterministic dedup (fuzzy name + matching
  ZIP, no LLM/network call involved).

The one live/networked end-to-end test lives in tests/test_pipeline.py.
"""

from __future__ import annotations

import pytest

from care_facilities import aco_match
from care_facilities import graph
from care_facilities.schema import Facility


# --- fetch_alf dedup helper ------------------------------------------------


def test_dedupe_by_npi_drops_exact_duplicates():
    facilities = [
        {"npi": "123", "name": "A"},
        {"npi": "456", "name": "B"},
        {"npi": "123", "name": "A"},  # duplicate, e.g. from overlapping NPPES pages
        {"npi": "123", "name": "A"},
        {"npi": "789", "name": "C"},
    ]
    deduped = graph._dedupe_by_npi(facilities)
    assert [f["npi"] for f in deduped] == ["123", "456", "789"]


def test_dedupe_by_npi_keeps_entries_with_no_npi():
    facilities = [{"npi": None, "name": "A"}, {"npi": None, "name": "B"}]
    assert graph._dedupe_by_npi(facilities) == facilities


# --- rank ---------------------------------------------------------------


def _facility(**overrides) -> Facility:
    base = dict(
        name="Test Facility",
        facility_type="SNF",
        address="123 Main St",
        city="Springfield",
        state="CA",
        zip="90001",
        distance_mi=5.0,
        cms_overall_rating=None,
        health_inspection_rating=None,
        staffing_rating=None,
        qm_rating=None,
        certified_beds=None,
        ownership_type=None,
        chain_name=None,
        affiliated_aco=None,
        leadership=None,
        phone=None,
        data_source="CMS",
        geocode_precision="exact",
    )
    base.update(overrides)
    return Facility(**base)


def test_rank_sorts_by_rating_desc_then_distance_asc():
    f_3star_far = _facility(name="3-star far", cms_overall_rating=3, distance_mi=10.0)
    f_5star_far = _facility(name="5-star far", cms_overall_rating=5, distance_mi=8.0)
    f_5star_near = _facility(name="5-star near", cms_overall_rating=5, distance_mi=1.0)
    f_none_near = _facility(name="unrated near", cms_overall_rating=None, distance_mi=0.5)
    f_1star = _facility(name="1-star", cms_overall_rating=1, distance_mi=2.0)

    state = {"facilities": [f_3star_far, f_none_near, f_5star_far, f_1star, f_5star_near]}
    result = graph.rank(state)
    ranked_names = [f.name for f in result["results"]]

    # 5-star facilities first (nearer one first), then 3-star, then 1-star,
    # then the unrated facility dead last -- even though it's the closest.
    assert ranked_names == [
        "5-star near",
        "5-star far",
        "3-star far",
        "1-star",
        "unrated near",
    ]


def test_rank_handles_empty_list():
    assert graph.rank({"facilities": []}) == {"results": []}


def test_rank_ties_on_rating_break_by_distance():
    a = _facility(name="a", cms_overall_rating=4, distance_mi=5.0)
    b = _facility(name="b", cms_overall_rating=4, distance_mi=1.0)
    c = _facility(name="c", cms_overall_rating=4, distance_mi=3.0)
    result = graph.rank({"facilities": [a, b, c]})
    assert [f.name for f in result["results"]] == ["b", "c", "a"]


# --- aco_match ------------------------------------------------------------


def _aco_row(aff_lbn: str, aco_name: str) -> dict:
    return {"Aff_LBN": aff_lbn, "ACO_Name": aco_name, "ACO_ID": "A1234"}


def test_aco_matcher_confident_exact_match():
    rows = [_aco_row("Golden Living Center LLC", "Golden Health ACO")]
    matcher = aco_match.AcoMatcher(rows)
    assert matcher.match("Golden Living Center") == "Golden Health ACO"
    assert matcher.match("GOLDEN LIVING CENTER, LLC") == "Golden Health ACO"


def test_aco_matcher_confident_fuzzy_match():
    rows = [_aco_row("Sunnyvale Rehabilitation and Nursing Center", "Sunnyvale ACO")]
    matcher = aco_match.AcoMatcher(rows)
    # Minor typo / near-exact match should still resolve confidently.
    assert matcher.match("Sunnyvale Rehabilitation and Nursing Centre") == "Sunnyvale ACO"


def test_aco_matcher_no_match_returns_none_never_fabricates():
    rows = [_aco_row("Golden Living Center LLC", "Golden Health ACO")]
    matcher = aco_match.AcoMatcher(rows)
    assert matcher.match("Totally Unrelated Nursing Facility Name") is None
    assert matcher.match("") is None
    assert matcher.match(None) is None


def test_aco_matcher_empty_dataset():
    matcher = aco_match.AcoMatcher([])
    assert matcher.match("Anything At All") is None


# --- reconcile: naive dedup -------------------------------------------------


def _snf_dict(**overrides) -> dict:
    base = dict(
        ccn="055555",
        name="Test SNF",
        address="1 Test Way",
        city="Testville",
        state="CA",
        zip="90001",
        latitude=34.0,
        longitude=-118.0,
        overall_rating=4,
        health_inspection_rating=4,
        staffing_rating=3,
        qm_rating=4,
        certified_beds=100,
        ownership_type="For profit - Corporation",
        chain_name="Test Chain",
        phone="5551234567",
        distance_mi=2.0,
        ownership=[],
        affiliated_aco=None,
    )
    base.update(overrides)
    return base


def _alf_dict(**overrides) -> dict:
    base = dict(
        npi="1234567890",
        name="Test ALF",
        address="2 Test Way",
        city="Testville",
        state="CA",
        zip="90001",
        phone="5559876543",
        leadership_name="Jane Doe",
        leadership_title="Administrator",
        latitude=34.01,
        longitude=-118.01,
        geocode_precision="address",
        distance_mi=3.0,
        bed_count=20,
    )
    base.update(overrides)
    return base


def test_reconcile_keeps_distinct_facilities_separate():
    state = {
        "snf_enriched": [_snf_dict()],
        "alf_enriched": [_alf_dict()],
        "errors": [],
    }
    result = graph.reconcile(state)

    facilities = result["facilities"]
    assert len(facilities) == 2
    types = {f.facility_type for f in facilities}
    assert types == {"SNF", "Assisted Living"}
    assert result["errors"] == []

    # Honesty guardrail holds with the naive dedup path too.
    alf = next(f for f in facilities if f.facility_type == "Assisted Living")
    assert alf.cms_overall_rating is None
    assert alf.affiliated_aco is None


def test_reconcile_merges_same_name_and_zip_across_sources():
    # Same ZIP, near-identical name (corp suffix stripped by normalize_name)
    # -> a continuing-care community that shows up once in CMS (SNF) and
    # once in NPPES (ALF); should merge, keeping the SNF entry (it carries
    # CMS ratings/ownership).
    state = {
        "snf_enriched": [_snf_dict(name="Sunnyvale Manor", zip="90001")],
        "alf_enriched": [_alf_dict(name="Sunnyvale Manor LLC", zip="90001")],
        "errors": [],
    }
    result = graph.reconcile(state)
    facilities = result["facilities"]
    assert len(facilities) == 1
    assert facilities[0].facility_type == "SNF"


def test_reconcile_does_not_merge_same_name_different_zip():
    state = {
        "snf_enriched": [_snf_dict(name="Golden Manor", zip="90001")],
        "alf_enriched": [_alf_dict(name="Golden Manor", zip="90099")],
        "errors": [],
    }
    result = graph.reconcile(state)
    assert len(result["facilities"]) == 2


def test_reconcile_does_not_merge_different_names_same_zip():
    state = {
        "snf_enriched": [_snf_dict(name="Golden Manor", zip="90001")],
        "alf_enriched": [_alf_dict(name="Completely Different Facility", zip="90001")],
        "errors": [],
    }
    result = graph.reconcile(state)
    assert len(result["facilities"]) == 2


def test_dedupe_facilities_prefers_more_complete_entry_within_same_type():
    sparse = Facility.from_snf_dict(
        _snf_dict(name="Golden Manor", zip="90001", ownership=[], chain_name=None)
    )
    complete = Facility.from_snf_dict(
        _snf_dict(
            name="Golden Manor",
            zip="90001",
            ownership=[{"owner_name": "Jane Doe", "role": "Administrator"}],
        )
    )
    deduped = graph._dedupe_facilities([sparse, complete])
    assert len(deduped) == 1
    assert deduped[0].leadership is not None
