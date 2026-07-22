"""Best-effort ACO (Accountable Care Organization) SNF-affiliation matching.

`sources/cms_snf.fetch_aco_affiliation` always returns `{}` because the real
CMS "ACO Skilled Nursing Facility Affiliates" dataset has no CCN/NPI field --
only a free-text `Aff_LBN` (affiliate legal business name) column -- so it
can't be joined to CCN-keyed facility data the way the ownership dataset can
(see the long comment above `fetch_aco_affiliation` in `sources/cms_snf.py`
for the full writeup of why).

This module implements the documented workaround: fetch the full ACO SNF
affiliates dataset live, cache it, and fuzzy-match each SNF facility's name
against `Aff_LBN` using conservative, high-cutoff matching -- in the same
spirit as `sources/stengel.match_bed_count`. This is a **best-effort
name-based match against a dataset that lacks a clean facility ID**, not an
authoritative CCN-based join like ownership uses. Treat any result as a
"probably" affiliation, not a certainty.

Honesty guardrail: if no confident match is found, the caller gets `None`,
never a guessed/nearest ACO. An ACO affiliation is a meaningful claim about a
facility's care-coordination arrangement -- fabricating or guess-attaching
one to satisfy a schema would be actively misleading, so this module would
rather under-match (report fewer affiliations than actually exist) than
over-match (report a wrong one).
"""

from __future__ import annotations

import difflib
from typing import Any

import httpx

from . import cache, config
from .sources import cms_snf

_REQUEST_TIMEOUT = 30.0
_PAGE_SIZE = 5000

# Same dataset ID/base CMS's cms_snf module already identified and documented
# as lacking a CCN/NPI field -- reused here rather than duplicated.
ACO_SNF_AFFILIATES_DATASET_ID = cms_snf.ACO_SNF_AFFILIATES_DATASET_ID
ACO_SNF_AFFILIATES_API_BASE = cms_snf.ACO_SNF_AFFILIATES_API_BASE

CACHE_KEY = "cms_aco_snf_affiliates_dataset"

# Fuzzy-match cutoff for difflib.get_close_matches -- deliberately high (per
# the task's guidance of 0.85+) since an ACO affiliation is a meaningful
# claim we don't want to guess our way into.
MATCH_CUTOFF = 0.87

_CORP_SUFFIXES = {
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "lp",
    "llp",
    "pllc",
    "pc",
}


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, and drop common corporate suffixes.

    Mirrors the spirit of `stengel._normalize_name` (alnum/whitespace-only,
    case-insensitive) plus suffix stripping so "Golden Living Center LLC"
    and "Golden Living Center" normalize identically.
    """
    text = "".join(ch for ch in str(name).lower() if ch.isalnum() or ch.isspace())
    tokens = [tok for tok in text.split() if tok not in _CORP_SUFFIXES]
    return " ".join(tokens).strip()


def fetch_aco_dataset() -> list[dict[str, Any]]:
    """Fetch the full "ACO SNF Affiliates" dataset (~3196 rows), paginating
    via `offset`/`size` against CMS's data-api/v1 API, and cache the result
    for `config.CACHE_TTL_CMS` seconds (7 days) since this dataset is
    updated on a slow (yearly/performance-year) cadence.

    Returns the raw list of row dicts (bare JSON list response, not a
    `{"results": [...]}` envelope -- confirmed live, see `cms_snf.py`'s
    module docstring for details on this API family).
    """

    def _fetch() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            response = httpx.get(
                f"{ACO_SNF_AFFILIATES_API_BASE}/{ACO_SNF_AFFILIATES_DATASET_ID}/data",
                params={"size": _PAGE_SIZE, "offset": offset},
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            page = response.json()
            if not isinstance(page, list) or not page:
                break
            rows.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return rows

    return cache.cached_call(CACHE_KEY, config.CACHE_TTL_CMS, _fetch)


class AcoMatcher:
    """Precomputed fuzzy-name index over the ACO SNF affiliates dataset.

    Build once per batch (via `load_aco_matcher()`) and reuse across many
    facilities -- this is the whole point of separating index-building from
    per-facility matching, since re-normalizing ~3196 rows per facility
    would be wasteful.
    """

    def __init__(self, aco_rows: list[dict[str, Any]]):
        self._normalized_names: list[str] = []
        self._name_to_aco: dict[str, str] = {}
        for row in aco_rows:
            aff_lbn = row.get("Aff_LBN")
            if not aff_lbn:
                continue
            aco_name = row.get("ACO_Name") or str(aff_lbn)
            normalized = _normalize_name(aff_lbn)
            if not normalized:
                continue
            if normalized not in self._name_to_aco:
                self._name_to_aco[normalized] = aco_name
                self._normalized_names.append(normalized)

    def match(self, facility_name: str | None) -> str | None:
        """Return the affiliated ACO's name for `facility_name`, or `None`
        if no confident match is found. Never guesses."""
        if not facility_name:
            return None
        normalized = _normalize_name(facility_name)
        if not normalized:
            return None

        # Exact match on normalized name first (preferred, per the task's
        # guidance to prefer exact/near-exact matches).
        if normalized in self._name_to_aco:
            return self._name_to_aco[normalized]

        close = difflib.get_close_matches(
            normalized, self._normalized_names, n=1, cutoff=MATCH_CUTOFF
        )
        if close:
            return self._name_to_aco.get(close[0])
        return None


def load_aco_matcher() -> AcoMatcher:
    """Fetch (or reuse the cached) ACO dataset and build an `AcoMatcher`."""
    return AcoMatcher(fetch_aco_dataset())
