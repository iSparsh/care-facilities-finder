"""Unified `Facility` output schema (Stage 3).

Both data sources (CMS SNF provider data and NPPES ALF registrations) are
normalized into this single Pydantic model so the rest of the pipeline
(reconciliation, ranking, the eventual FastAPI/HTML UI) only has to deal
with one shape.

Honesty guardrail: `cms_overall_rating` and `affiliated_aco` are semantically
"not available for this facility type / not found", never a fabricated
guess. In particular, Assisted Living facilities ALWAYS get
`cms_overall_rating=None` and `affiliated_aco=None` -- CMS star ratings and
ACO affiliations are SNF/Medicare concepts that simply do not apply to
NPPES-registered ALFs. `from_alf_dict` hardcodes both to `None` rather than
leaving them to be filled in by any enrichment/reconciliation step, so no
downstream code (including the LLM reconcile node in `graph.py`) can
accidentally fabricate a rating or ACO for an ALF.

`geocode_precision` convention:
- `"exact"` -- SNF facilities always get this (CMS's provider-info dataset
  publishes real lat/lon per facility), and ALF facilities get this when the
  US Census address geocoder found a real street-address match (surfaced by
  `nppes_alf.geocode_alf_facilities` as `geocode_precision == "address"`).
- `"zip_centroid"` -- ALF facilities that fell back to a ZIP-code centroid
  because the Census geocoder had no address match.
- `None` -- no coordinates could be resolved at all (should be rare/never in
  practice since un-geocodable facilities are dropped by the radius filter
  upstream, but modeled here defensively).
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

FacilityType = Literal["SNF", "Assisted Living"]


class Facility(BaseModel):
    """A single elder-care facility, normalized from either CMS (SNF) or
    NPPES (ALF) source data."""

    name: str
    facility_type: FacilityType
    address: str
    city: str
    state: str
    zip: str
    distance_mi: float

    cms_overall_rating: Optional[int] = None
    health_inspection_rating: Optional[int] = None
    staffing_rating: Optional[int] = None
    qm_rating: Optional[int] = None
    certified_beds: Optional[int] = None

    ownership_type: Optional[str] = None
    chain_name: Optional[str] = None
    affiliated_aco: Optional[str] = None
    leadership: Optional[str] = None
    phone: Optional[str] = None

    data_source: str
    geocode_precision: Optional[str] = None

    @classmethod
    def from_snf_dict(cls, facility: dict[str, Any]) -> "Facility":
        """Build a `Facility` from a post-enrichment CMS SNF facility dict.

        Expects the raw `cms_snf.fetch_snf_facilities` / `filter_by_radius`
        keys, plus enrichment keys attached by `graph.enrich_snf`:
        `ownership` (list of owner/manager dicts, see
        `cms_snf.fetch_ownership`) and `affiliated_aco` (str | None, see
        `aco_match.py`).
        """
        return cls(
            name=facility.get("name") or "",
            facility_type="SNF",
            address=facility.get("address") or "",
            city=facility.get("city") or "",
            state=facility.get("state") or "",
            zip=facility.get("zip") or "",
            distance_mi=float(facility.get("distance_mi") or 0.0),
            cms_overall_rating=facility.get("overall_rating"),
            health_inspection_rating=facility.get("health_inspection_rating"),
            staffing_rating=facility.get("staffing_rating"),
            qm_rating=facility.get("qm_rating"),
            certified_beds=facility.get("certified_beds"),
            ownership_type=facility.get("ownership_type"),
            chain_name=facility.get("chain_name"),
            affiliated_aco=facility.get("affiliated_aco"),
            leadership=_format_snf_leadership(facility.get("ownership")),
            phone=facility.get("phone"),
            data_source="CMS",
            geocode_precision="exact",
        )

    @classmethod
    def from_alf_dict(cls, facility: dict[str, Any]) -> "Facility":
        """Build a `Facility` from a post-enrichment NPPES ALF facility dict.

        Expects the raw `nppes_alf.fetch_alf_facilities` /
        `geocode_alf_facilities` / `filter_by_radius` keys, plus the
        `bed_count` enrichment key attached by `graph.enrich_alf` (via
        `stengel.match_bed_count`, may be `None`).

        `cms_overall_rating` and `affiliated_aco` are always `None` here --
        see the module docstring's honesty guardrail.
        """
        return cls(
            name=facility.get("name") or "",
            facility_type="Assisted Living",
            address=facility.get("address") or "",
            city=facility.get("city") or "",
            state=facility.get("state") or "",
            zip=facility.get("zip") or "",
            distance_mi=float(facility.get("distance_mi") or 0.0),
            cms_overall_rating=None,
            health_inspection_rating=None,
            staffing_rating=None,
            qm_rating=None,
            certified_beds=facility.get("bed_count"),
            ownership_type=None,
            chain_name=None,
            affiliated_aco=None,
            leadership=_format_alf_leadership(facility),
            phone=facility.get("phone"),
            data_source="NPPES",
            geocode_precision=_map_alf_geocode_precision(
                facility.get("geocode_precision")
            ),
        )


# --- formatting helpers ------------------------------------------------------


def _format_owner(owner: dict[str, Any]) -> Optional[str]:
    name = owner.get("owner_name")
    if not name:
        return None
    details = []
    role = owner.get("role")
    if role:
        details.append(str(role))
    pct = owner.get("ownership_percentage")
    if pct is not None:
        details.append(f"{pct:g}%")
    if details:
        return f"{name} ({', '.join(details)})"
    return str(name)


def _format_snf_leadership(ownership: Optional[list[dict[str, Any]]]) -> Optional[str]:
    if not ownership:
        return None
    formatted = [text for owner in ownership if (text := _format_owner(owner))]
    return "; ".join(formatted) if formatted else None


def _format_alf_leadership(facility: dict[str, Any]) -> Optional[str]:
    name = facility.get("leadership_name")
    title = facility.get("leadership_title")
    if name and title:
        return f"{name} ({title})"
    return name or title or None


def _map_alf_geocode_precision(raw: Optional[str]) -> Optional[str]:
    if raw == "address":
        return "exact"
    if raw == "zip_centroid":
        return "zip_centroid"
    return None
