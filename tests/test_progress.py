"""Tests for ZIP-centroid ALF prefiltering and progress emission."""

from __future__ import annotations

from care_facilities import progress
from care_facilities.sources import nppes_alf


def test_prefilter_keeps_nearby_zips_only():
    # Origin near 94404 (San Mateo / Foster City area).
    origin_lat, origin_lon = 37.5538, -122.27
    facilities = [
        {"name": "near", "zip": "94404"},
        {"name": "also-near", "zip": "94010"},  # Burlingame-ish
        {"name": "far", "zip": "92101"},  # San Diego
        {"name": "unknown-zip", "zip": ""},  # kept (can't exclude)
    ]
    kept = nppes_alf.prefilter_by_zip_centroid(
        facilities, origin_lat, origin_lon, radius_miles=10, buffer_miles=15
    )
    names = {f["name"] for f in kept}
    assert "near" in names
    assert "also-near" in names
    assert "unknown-zip" in names
    assert "far" not in names


def test_progress_emit_invokes_callback():
    events = []
    with progress.progress_scope(events.append):
        progress.emit("demo", "hello", count=1)
    assert events == [{"stage": "demo", "message": "hello", "count": 1}]


def test_progress_emit_noop_without_callback():
    progress.emit("demo", "should not raise")
