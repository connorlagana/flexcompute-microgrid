"""Experiment B's two new constraints: reliability matching and fixed duration.

The earlier Experiment B compared a re-sized fixed-load plant that browned out
for hundreds of MWh a year against flexible designs that browned out for none,
and called the capital difference the value of flexible operation. Some of it
was the value of *lower reliability*. B2 removes that by capping involuntary
shortfall identically for everyone.

The tests below are about the search mechanics rather than the economics: that
the cap is actually enforced, that it is never enforced in a way that quietly
favours flexibility, that pinning duration really pins it, and that a duration
outside the cost source's range is flagged as an extrapolation rather than
quoted as a price.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.costs import from_upstream_config
from flexcompute.experiments import (
    COST_SOURCE_DURATION_RANGE_H,
    Sizing,
    SizingSearchResult,
    SizingSearchSpec,
    duration_is_extrapolated,
    minimum_capex_for_compute,
    run_sizing_searches,
    run_strategy,
)
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot

from conftest import BASELINE_SCENARIO, requires_weather

SNAPSHOT_PATH = SNAPSHOT_DIR / f"{BASELINE_SCENARIO.label()}.json"

requires_snapshot = pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(), reason="run scripts/run_baseline.py first"
)


@pytest.fixture(scope="module")
def reference():
    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    return Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"])


# ---------------------------------------------------------------------------
# Duration extrapolation labelling
# ---------------------------------------------------------------------------

def test_the_cost_sources_duration_range_is_what_nrel_published():
    """2-10 h. Outside that the $/kW + $/kWh split is not sourced."""
    assert COST_SOURCE_DURATION_RANGE_H == (2.0, 10.0)


@pytest.mark.parametrize(
    "duration, extrapolated",
    [(1.0, True), (2.0, False), (4.0, False), (10.0, False), (14.0, True), (15.1, True)],
)
def test_durations_outside_the_source_range_are_flagged(duration, extrapolated):
    assert duration_is_extrapolated(duration) is extrapolated


def test_a_result_carries_its_extrapolation_flag():
    """The flag has to travel with the number, not live in a footnote.

    The optimiser has repeatedly chosen ~14-15 hour storage. That may well be
    right, but the cost model cannot price it from its source data, and a
    reader of the result must be told so without having to know the range.
    """
    result = SizingSearchResult(
        strategy="x", sizing=Sizing(100.0, 40.0, 15.0), capex_musd=1.0,
        compute_units=1.0, target_compute=1.0, involuntary_shortfall_mwh=0.0,
        breakdown={}, evaluations=1, wall_time_s=0.0, success=True,
    )
    assert result.duration_extrapolated
    assert result.as_dict()["duration_extrapolated"] is True
    assert result.as_dict()["cost_source_duration_range_h"] == [2.0, 10.0]


def test_an_in_range_duration_is_not_flagged():
    result = SizingSearchResult(
        strategy="x", sizing=Sizing(100.0, 40.0, 8.0), capex_musd=1.0,
        compute_units=1.0, target_compute=1.0, involuntary_shortfall_mwh=0.0,
        breakdown={}, evaluations=1, wall_time_s=0.0, success=True,
    )
    assert not result.duration_extrapolated


# ---------------------------------------------------------------------------
# The searches themselves
# ---------------------------------------------------------------------------

@requires_weather
@requires_snapshot
def test_fixed_duration_is_actually_held_fixed(site, reference):
    """Task 7 depends on this: a 'fixed 8 h' row must really be 8 h."""
    cost_model = from_upstream_config(site.config, site.scenario.architecture)
    target = run_strategy(site, "fixed_load", reference.scaled(0.5)).metrics["compute_units"]

    result = minimum_capex_for_compute(
        site, "fixed_load", target_compute=target, cost_model=cost_model,
        reference=reference, fixed_duration_h=8.0, maxiter=3, popsize=3,
    )
    assert result.sizing.duration_h == pytest.approx(8.0)
    assert result.duration_mode == "fixed_8h"
    assert result.sizing.battery_mwh == pytest.approx(8.0 * result.sizing.battery_mw)


@requires_weather
@requires_snapshot
def test_free_duration_reports_itself_as_free(site, reference):
    cost_model = from_upstream_config(site.config, site.scenario.architecture)
    target = run_strategy(site, "fixed_load", reference.scaled(0.5)).metrics["compute_units"]
    result = minimum_capex_for_compute(
        site, "fixed_load", target_compute=target, cost_model=cost_model,
        reference=reference, maxiter=3, popsize=3,
    )
    assert result.duration_mode == "free"


@requires_weather
@requires_snapshot
def test_the_reliability_cap_is_enforced(site, reference):
    """B2's whole point: a design may not buy compute with brownouts.

    Run the same search with and without a tight cap. The constrained result
    must respect it; the unconstrained one is free not to.
    """
    cost_model = from_upstream_config(site.config, site.scenario.architecture)
    scarce = reference.scaled(0.45)
    target = run_strategy(site, "fixed_load", scarce).metrics["compute_units"]
    cap = 50.0

    constrained = minimum_capex_for_compute(
        site, "fixed_load", target_compute=target, cost_model=cost_model,
        reference=reference, max_shortfall_mwh=cap, maxiter=6, popsize=6,
    )
    assert constrained.max_shortfall_mwh == cap
    assert constrained.reliability_met
    assert constrained.involuntary_shortfall_mwh <= cap + 1e-6


@requires_weather
@requires_snapshot
def test_reliability_matching_never_makes_a_design_cheaper(site, reference):
    """A constraint can only cost capital, never save it.

    If B2 ever came out cheaper than B1 for the same strategy, the constraint
    would be doing something other than constraining -- most likely the search
    landing somewhere different by luck. Both searches use the same seed, so
    the comparison is meaningful.
    """
    cost_model = from_upstream_config(site.config, site.scenario.architecture)
    scarce = reference.scaled(0.45)
    target = run_strategy(site, "fixed_load", scarce).metrics["compute_units"]

    common = dict(
        target_compute=target, cost_model=cost_model, reference=reference,
        maxiter=6, popsize=6, seed=7,
    )
    b1 = minimum_capex_for_compute(site, "fixed_load", **common)
    b2 = minimum_capex_for_compute(site, "fixed_load", max_shortfall_mwh=25.0, **common)

    assert b2.capex_musd >= b1.capex_musd - 1e-6


@requires_weather
@requires_snapshot
def test_the_cap_costs_nothing_to_a_design_that_was_already_reliable(site, reference):
    """The criterion must not be one that favours flexible operation by design.

    A strategy whose unconstrained optimum already sits inside the cap should
    return exactly the same design when the cap is applied. If it did not, the
    constraint would be penalising strategies for reasons unrelated to
    reliability.
    """
    cost_model = from_upstream_config(site.config, site.scenario.architecture)
    scarce = reference.scaled(0.45)
    target = run_strategy(site, "fixed_load", scarce).metrics["compute_units"]

    common = dict(
        target_compute=target, cost_model=cost_model, reference=reference,
        maxiter=5, popsize=5, seed=11,
    )
    unconstrained = minimum_capex_for_compute(site, "casey_governor", **common)
    generous_cap = unconstrained.involuntary_shortfall_mwh + 1000.0
    constrained = minimum_capex_for_compute(
        site, "casey_governor", max_shortfall_mwh=generous_cap, **common
    )

    assert constrained.capex_musd == pytest.approx(unconstrained.capex_musd, rel=1e-9)
    assert constrained.sizing.solar_mw == pytest.approx(unconstrained.sizing.solar_mw)


@requires_weather
@requires_snapshot
def test_every_search_meets_the_compute_target_it_claims(site, reference):
    """A design that misses the target is not an answer, and must say so."""
    cost_model = from_upstream_config(site.config, site.scenario.architecture)
    target = run_strategy(site, "fixed_load", reference.scaled(0.5)).metrics["compute_units"]
    result = minimum_capex_for_compute(
        site, "casey_governor", target_compute=target, cost_model=cost_model,
        reference=reference, maxiter=5, popsize=5,
    )
    if result.success:
        assert result.compute_units >= target * (1 - 1e-6)


# ---------------------------------------------------------------------------
# The parallel harness
# ---------------------------------------------------------------------------

@requires_weather
@requires_snapshot
def test_parallel_and_serial_sizing_searches_agree(reference):
    """Searches are independent, so splitting them must change nothing."""
    specs = [
        SizingSearchSpec(
            scenario=BASELINE_SCENARIO, strategy=strategy,
            target_compute=6000.0, reference=reference, label=strategy,
            maxiter=3, popsize=3,
        )
        for strategy in ("fixed_load", "casey_governor")
    ]
    serial = run_sizing_searches(specs, workers=1)
    parallel = run_sizing_searches(specs, workers=2)

    assert set(serial) == set(parallel)
    for label in serial:
        assert serial[label].capex_musd == pytest.approx(
            parallel[label].capex_musd, rel=1e-9
        )
        assert serial[label].sizing.solar_mw == pytest.approx(
            parallel[label].sizing.solar_mw
        )
