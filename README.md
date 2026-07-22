# care-facilities-finder

Given a US zipcode, finds nearby elder-care facilities (nursing homes and
assisted living) and returns a ranked table with CMS star ratings, ACO
affiliation, and leadership/ownership info.

Data comes from CMS Care Compare, NPPES, and the US Census Geocoder.
Duplicate SNF/ALF entries are merged with a deterministic fuzzy-name + ZIP
match — no LLM required. Orchestrated with
[LangGraph](https://github.com/langchain-ai/langgraph); served via FastAPI +
a static HTML/JS UI (`api/`).

## Setup

```bash
# from the repository root
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Optional: set APP_USERNAME + APP_PASSWORD to exercise Basic Auth locally.
# Leave unset (or blank) to skip the login prompt. REDIS_URL is optional;
# unset uses a local SQLite cache under .cache/.
```

## Running the web app

```bash
# from the repository root, with the venv activated
uvicorn api.main:app --reload --app-dir .
```

Open `http://localhost:8000`. `GET /health` is a liveness check; `POST /search`
accepts `{"zipcode": "94404", "radius_miles": 15}`.

Cold searches can take several minutes (Census geocoding per ALF address);
warm cache hits are much faster.

## Tests

```bash
pytest
```

## Data sources and known limitations

- **CMS Provider Data** — SNF info, star ratings, ownership.
- **NPPES NPI Registry** — ALF name/address/leadership (no federal ALF feed).
- **US Census Geocoder** — ALF lat/lon (ZIP-centroid fallback when needed).
- **Stengel dataset** — best-effort ALF bed-count enrichment.
- **ALFs have no CMS ratings or ACO affiliation** — those are Medicare/SNF
  concepts; the API/UI always show `null` / "N/A" for ALF rows, never a
  fabricated value.
- **ACO matching is best-effort and low-recall** — fuzzy name match at a high
  cutoff (0.87); prefers under-matching over a wrong affiliation.
