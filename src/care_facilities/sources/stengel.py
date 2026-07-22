"""Optional bed-count enrichment from the Anton Stengel assisted-living
dataset (https://github.com/antonstengel/assisted-living-data).

That repo publishes a static, ~44,638-row CSV snapshot (compiled ~2022 from
state licensing data) of US assisted-living facilities, including a
"Capacity" column (bed count) that NPPES has no equivalent for. This module
is best-effort enrichment, not a hard dependency -- if the download fails
for any reason, `load_stengel_dataset` logs a warning and returns None, and
callers should treat missing bed counts as normal/expected.

Caching strategy: this is a large (~13 MB) *static* file that rarely (if
ever) changes, so rather than going through `cache.cached_call` (which
round-trips values through `json.dumps` in sqlite -- wasteful for a 13MB
CSV blob), we save the raw CSV bytes to a plain file under the project's
`.cache/` directory and treat it as fresh for `STENGEL_CACHE_TTL` seconds
based on the file's mtime. This is simpler to inspect/debug (it's just a
CSV on disk) and avoids bloating cache.sqlite3.
"""

from __future__ import annotations

import difflib
import time
import warnings
from pathlib import Path

import httpx
import pandas as pd

from .. import config

STENGEL_CSV_URL = (
    "https://raw.githubusercontent.com/antonstengel/assisted-living-data"
    "/main/assisted-living-facilities.csv"
)

# Static 2022 snapshot -- cache effectively indefinitely (180 days), since
# there's no expectation of it changing on the upstream repo.
STENGEL_CACHE_TTL = 180 * 24 * 3600

STENGEL_CACHE_FILENAME = "stengel_assisted_living_facilities.csv"


def _stengel_cache_path() -> Path:
    return config.cache_dir_path() / STENGEL_CACHE_FILENAME


def load_stengel_dataset() -> pd.DataFrame | None:
    """Load the Stengel assisted-living CSV, downloading/caching as needed.

    Returns None (with a warning logged) if the download fails and there's
    no usable cached copy on disk -- this dataset is optional enrichment,
    not a hard dependency.
    """
    cache_path = _stengel_cache_path()

    if cache_path.exists():
        age_seconds = time.time() - cache_path.stat().st_mtime
        if age_seconds < STENGEL_CACHE_TTL:
            try:
                return _read_csv(cache_path)
            except Exception as exc:  # corrupt cache file; try a re-download
                warnings.warn(
                    f"stengel: cached CSV at {cache_path} unreadable ({exc}); "
                    "attempting re-download."
                )

    try:
        response = httpx.get(STENGEL_CSV_URL, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
        return _read_csv(cache_path)
    except Exception as exc:
        warnings.warn(
            f"stengel: failed to download assisted-living dataset from "
            f"{STENGEL_CSV_URL}: {exc}. Bed-count enrichment will be skipped."
        )
        # Fall back to a stale on-disk copy, if any, rather than nothing.
        if cache_path.exists():
            try:
                return _read_csv(cache_path)
            except Exception:
                pass
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"Zip Code": str}, low_memory=False)


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum() or ch.isspace()).strip()


def _zip5(value: object) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits.zfill(5)[:5] if digits else ""


def match_bed_count(facility: dict, df: pd.DataFrame | None) -> int | None:
    """Best-effort match of `facility` (an NPPES ALF dict) against the
    Stengel dataframe, returning its bed count ("Capacity") if found.

    Matching approach (intentionally simple, not a fuzzy-matching library):
    1. Narrow candidates to rows sharing the facility's 5-digit ZIP; if none,
       fall back to rows sharing the facility's city (case-insensitive).
    2. Look for an exact/substring match on normalized (lowercased,
       alnum-only) facility name.
    3. Otherwise, try `difflib.get_close_matches` on normalized names among
       the narrowed candidates.

    Returns None if `df` is None/empty or no reasonable match is found. This
    is deliberately conservative -- a missed match is preferable to
    attaching the wrong facility's bed count.
    """
    if df is None or df.empty:
        return None

    name_norm = _normalize_name(facility.get("name", ""))
    if not name_norm:
        return None

    zip5 = _zip5(facility.get("zip", ""))
    candidates = pd.DataFrame()
    if zip5:
        candidates = df[df["Zip Code"].map(_zip5) == zip5]

    if candidates.empty:
        city = str(facility.get("city", "")).strip().lower()
        if city:
            candidates = df[df["City"].astype(str).str.strip().str.lower() == city]

    if candidates.empty:
        return None

    names = candidates["Facility Name"].astype(str).tolist()
    normalized_names = [_normalize_name(n) for n in names]

    for idx, cand_norm in enumerate(normalized_names):
        if cand_norm and (
            cand_norm == name_norm or cand_norm in name_norm or name_norm in cand_norm
        ):
            return _extract_capacity(candidates.iloc[idx])

    close = difflib.get_close_matches(name_norm, normalized_names, n=1, cutoff=0.72)
    if close:
        idx = normalized_names.index(close[0])
        return _extract_capacity(candidates.iloc[idx])

    return None


def _extract_capacity(row: pd.Series) -> int | None:
    value = row.get("Capacity")
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
