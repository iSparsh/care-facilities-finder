"""Entry point for the Stage 3 pipeline: zipcode -> ranked `Facility` list.

Usage as a library:

    from care_facilities.pipeline import run
    results = run("94404", radius_miles=15)

Usage from the command line:

    python -m care_facilities.pipeline 94404 15
"""

from __future__ import annotations

import sys

from . import config
from .graph import COMPILED_GRAPH, PipelineState
from .schema import Facility


def run(zipcode: str, radius_miles: float | None = None) -> list[Facility]:
    """Run the full pipeline for `zipcode` and return the ranked results.

    `radius_miles` defaults to `config.DEFAULT_RADIUS_MILES` when omitted.
    Never raises for "expected" failure modes (bad zipcode, network errors)
    -- those degrade to an empty result list, with details recorded in the
    pipeline's internal `errors` list (not currently surfaced here, but
    visible via `COMPILED_GRAPH.invoke` directly if needed for debugging).
    """
    radius = float(radius_miles) if radius_miles is not None else float(
        config.DEFAULT_RADIUS_MILES
    )
    initial_state: PipelineState = {
        "zipcode": zipcode,
        "radius_miles": radius,
        "errors": [],
    }
    final_state = COMPILED_GRAPH.invoke(initial_state)
    return final_state.get("results") or []


def _format_table(results: list[Facility]) -> str:
    if not results:
        return "No facilities found."

    fields = list(Facility.model_fields.keys())
    rows: list[dict[str, str]] = []
    for facility in results:
        row = {}
        for name in fields:
            value = getattr(facility, name)
            row[name] = "N/A" if value is None else str(value)
        rows.append(row)

    max_col_width = 28
    widths = {
        name: min(max(len(name), max(len(row[name]) for row in rows)), max_col_width)
        for name in fields
    }

    def fmt_row(row: dict[str, str]) -> str:
        cells = []
        for name in fields:
            text = row[name]
            if len(text) > widths[name]:
                text = text[: widths[name] - 1] + "…"
            cells.append(text.ljust(widths[name]))
        return " | ".join(cells)

    header = fmt_row({name: name for name in fields})
    separator = "-+-".join("-" * widths[name] for name in fields)

    lines = [header, separator]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


if __name__ == "__main__":
    arg_zipcode = sys.argv[1] if len(sys.argv) > 1 else "94404"
    arg_radius = float(sys.argv[2]) if len(sys.argv) > 2 else config.DEFAULT_RADIUS_MILES

    print(f"Searching for elder-care facilities within {arg_radius} miles of {arg_zipcode}...\n")
    facilities = run(arg_zipcode, arg_radius)
    print(_format_table(facilities))
    print(f"\n{len(facilities)} facilities found.")
