"""The reproducibility contract.

Re-runs the baseline scenario and demands bit-level agreement with the stored
snapshot. This is the test that makes the Phase 2 invariant enforceable:
*introducing a controller abstraction must not move the fixed-load numbers.*
"""

from __future__ import annotations

import pytest

from flexcompute.snapshot import (
    SNAPSHOT_DIR,
    build_snapshot,
    compare_snapshots,
    load_snapshot,
)

from conftest import BASELINE_SCENARIO, requires_weather

SNAPSHOT_PATH = SNAPSHOT_DIR / f"{BASELINE_SCENARIO.label()}.json"

requires_snapshot = pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(),
    reason=f"No stored baseline at {SNAPSHOT_PATH}. Run scripts/run_baseline.py first.",
)


@pytest.fixture(scope="module")
def stored():
    return load_snapshot(SNAPSHOT_PATH)


@requires_weather
@requires_snapshot
def test_site_construction_is_reproducible(site, stored):
    """Weather, cooling selection, PUE and load profile all reproduce."""
    fresh = site.provenance()
    diffs = compare_snapshots(stored["provenance"], fresh)
    assert diffs == [], "\n".join(diffs)


@requires_weather
@requires_snapshot
def test_fixed_load_dispatch_is_reproducible(site, stored):
    """Dispatch at every probed sizing reproduces exactly."""
    fresh = build_snapshot(site, run_optimizer=False)
    diffs = compare_snapshots(
        {k: v for k, v in stored.items() if k != "optimized"}, fresh
    )
    assert diffs == [], "\n".join(diffs)


@requires_weather
@requires_snapshot
def test_sizing_optimization_is_reproducible(site, stored):
    """The seeded sizing search lands on the same design and the same cost."""
    fresh = build_snapshot(site, run_optimizer=True)
    diffs = compare_snapshots(stored, fresh)
    assert diffs == [], "\n".join(diffs)


@requires_weather
@requires_snapshot
def test_snapshot_records_a_clean_audit(stored):
    """Whatever else changes, the stored baseline must be physically valid."""
    from flexcompute.metrics import EnergyAudit

    for probe in stored["fixed_sizing_probes"]:
        assert EnergyAudit(**probe["audit"]).violations() == []
    if "optimized" in stored:
        assert EnergyAudit(**stored["optimized"]["audit"]).violations() == []


@requires_snapshot
def test_snapshot_declares_its_weather_source(stored):
    """Absolute results are only comparable within one weather source."""
    assert stored["provenance"]["weather"]["source"] in {"pvgis", "nsrdb"}
    assert stored["provenance"]["scenario"]["seed"] is not None
