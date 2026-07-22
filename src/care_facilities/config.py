"""Central configuration for the care_facilities project.

Later stages (data-source modules, the LangGraph pipeline, the FastAPI UI)
should import shared constants and settings from this module rather than
hard-coding them.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file (if present) into the process environment.
load_dotenv()

# --- Search defaults -------------------------------------------------------

DEFAULT_RADIUS_MILES = 25

# --- CMS Provider Data API ---------------------------------------------------
# https://data.cms.gov/provider-data/

CMS_API_BASE = "https://data.cms.gov/provider-data/api/1/datastore/query"

# Skilled Nursing Facility / Nursing Home provider info + star ratings dataset.
CMS_SNF_PROVIDER_INFO_DATASET = "4pq5-n9py"

# Skilled Nursing Facility ownership dataset.
CMS_SNF_OWNERSHIP_DATASET = "y2hd-n93e"

# --- NPPES NPI Registry API ---------------------------------------------------

NPPES_API_BASE = "https://npiregistry.cms.hhs.gov/api/"
NPPES_API_VERSION = "2.1"

# --- US Census Geocoder --------------------------------------------------

CENSUS_GEOCODER_BASE = "https://geocoding.geo.census.gov/geocoder"

# --- Disk cache --------------------------------------------------------------

# Relative to the current working directory (the project root, in normal use).
CACHE_DIR = ".cache"

CACHE_TTL_CMS = 7 * 24 * 3600
CACHE_TTL_NPPES = 7 * 24 * 3600
CACHE_TTL_GEOCODE = 90 * 24 * 3600

# --- Claude / LangGraph --------------------------------------------------

CLAUDE_MODEL = "claude-sonnet-5"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def cache_dir_path() -> Path:
    """Return CACHE_DIR as a Path (relative to the current working directory)."""
    return Path(CACHE_DIR)
