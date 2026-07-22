# care-facilities

Given a US zipcode, `care-facilities-finder` finds nearby elder-care facilities
(nursing homes and assisted living communities) and returns a ranked table
enriched with:

- CMS star ratings (quality, staffing, health inspections)
- ACO (Accountable Care Organization) affiliation
- Facility leadership / ownership info

The pipeline is orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph)
and pulls data from public sources (CMS Care Compare, NPPES, US Census
Geocoder). Duplicate SNF/ALF entries are merged with a deterministic
fuzzy-name + ZIP match — no LLM required.

## Status

**Complete, end-to-end.** All four stages are built and tested:

- Stage 1: configuration (`src/care_facilities/config.py`), a disk-backed
  HTTP response cache (`src/care_facilities/cache.py`), and zipcode/address
  geocoding utilities (`src/care_facilities/geocode.py`)
- Stage 2: data-source modules (`src/care_facilities/sources/cms_snf.py`,
  `nppes_alf.py`, `stengel.py`) and ACO name-matching
  (`src/care_facilities/aco_match.py`)
- Stage 3: the LangGraph pipeline (`src/care_facilities/graph.py`,
  `pipeline.py`) that fetches, enriches, reconciles, and ranks facilities
- Stage 4: a FastAPI backend + static HTML/JS UI (`api/main.py`,
  `api/static/index.html`) for entering a zipcode and browsing results in a
  browser

## Setup

```bash
# from the repository root
pip install -e ".[dev]"

cp .env.example .env
# Optional for local use. Leave APP_USERNAME / APP_PASSWORD unset to run
# without a login prompt. Set both to exercise Basic Auth locally.
```

## Running the web app

```bash
source .venv/bin/activate   # or call .venv/bin/uvicorn directly, no activate needed
uvicorn api.main:app --reload --app-dir .
```

Then open `http://localhost:8000` in a browser. `GET /health` is a plain
liveness check, and `POST /search` (used by the UI) accepts
`{"zipcode": "94404", "radius_miles": 15}` and returns the ranked facility
list as JSON.

**Note on timing:** a search performs live calls to the CMS Provider Data
API, the NPPES NPI Registry, and (for any not-yet-cached assisted-living
addresses) the US Census geocoder. A warm cache (a zip/state searched
before) responds in seconds; a fully cold cache in a state with many
uncached ALF addresses can take several minutes, since each address is
geocoded individually (and then cached for next time -- repeat searches in
the same state get progressively faster). The UI's loading state says this
up front rather than looking hung.

## Running tests

```bash
pytest
# or just the Stage 4 API tests:
pytest tests/test_api.py -v
```

## Data sources and known limitations

- **CMS Provider Data** (Care Compare "Nursing Home" datasets): skilled
  nursing facility (SNF) info, star ratings, and ownership.
- **NPPES NPI Registry**: assisted living facility (ALF) organizational
  registrations (name/address/leadership), since ALFs are state-licensed
  and have no equivalent federal provider-data feed.
- **US Census Geocoder**: turns ALF street addresses into lat/lon (with a
  ZIP-centroid fallback when an exact address match isn't found).
- **Stengel assisted-living dataset**: best-effort bed-count enrichment for
  ALFs, joined by fuzzy name/address match.
- **Lack of CMS ratings in some records**: assisted living facilities have no
  federal CMS star rating and no ACO (Accountable Care Organization)
  affiliation. CMS star ratings and ACOs are Medicare/SNF-specific concepts
  that simply do not exist for ALFs under current US elder-care regulation.
  Both the API and the UI always show these fields as `null` / "N/A" for ALF
  rows -- never a fabricated or guessed value (see the docstring in `src/care_facilities/schema.py` for the full guardrail rationale).
- **ACO matching is best-effort, name-based, and low-recall by design.** The
CMS "ACO SNF Affiliates" dataset has no CCN/NPI join key, only a free-text
legal-business-name column, so `aco_match.py` fuzzy-matches SNF names
against it with a deliberately high similarity cutoff (0.87). It would
rather under-match (report no affiliation) than over-match (report a wrong
one), so seeing `affiliated_aco: null` for most SNFs is expected and normal
