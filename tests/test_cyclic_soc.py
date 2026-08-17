"""Cyclic state-of-charge boundary condition (ASSUMPTIONS Q1).

Upstream hands the simulation a 75%-full battery for free and never asks for it
back. The cyclic mode requires the year to be self-sustaining instead. These
tests check that the fixed point actually converges, that it removes the free
energy, and that the fixed-mode default is untouched so the Milestone 1 gate
still holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.control import FixedLoadController, SimpleThrottleController
from flexcompute.dispatch import UPSTREAM_INITIAL_SOC_PCT, simulate, simulate_cyclic
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot

from conftest import BASELINE_SCENARIO, requires_weather

SNAPSHOT_PATH = SNAPSHOT_DIR / f"{BASELINE_SCENARIO.label()}.json"

requires_snapshot = pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(), reason="run scripts/run_baseline.py first"
)


@pytest.fixture(scope="module")
def sizing():
    if not SNAPSHOT_PATH.exists():
        pytest.skip("no baseline snapshot")
    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    return s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"]


@requires_weather
@requires_snapshot
@pytest.mark.parametrize(
    "controller_factory", [FixedLoadController, SimpleThrottleController]
)
def test_cyclic_year_is_self_sustaining(site, sizing, controller_factory):
    solar_mw, battery_mw, duration_h = sizing
    run = simulate_cyclic(
        site, controller_factory(), solar_mw=solar_mw,
        battery_mw=battery_mw, battery_duration_h=duration_h,
    )
    info = run.metadata["cyclic_soc"]
    assert info["converged"]
    assert abs(run.metrics["soc_end_mwh"] - run.metrics["soc_start_mwh"]) <= info["tolerance_mwh"]
    assert run.metrics["soc_net_change_mwh"] == pytest.approx(0.0, abs=info["tolerance_mwh"])


@requires_weather
@requires_snapshot
def test_cyclic_removes_the_free_starting_energy(site, sizing):
    """The fixed-mode run ends below where it started; the cyclic one does not."""
    solar_mw, battery_mw, duration_h = sizing
    fixed = simulate(
        site, FixedLoadController(), solar_mw=solar_mw,
        battery_mw=battery_mw, battery_duration_h=duration_h,
    )
    assert fixed.metrics["soc_net_change_mwh"] < 0.0     # consumed unearned energy

    cyclic = simulate_cyclic(
        site, FixedLoadController(), solar_mw=solar_mw,
        battery_mw=battery_mw, battery_duration_h=duration_h,
    )
    assert cyclic.metrics["soc_net_change_mwh"] == pytest.approx(0.0, abs=1e-6)
    assert cyclic.metrics["soc_start_mwh"] < fixed.metrics["soc_start_mwh"]


@requires_weather
@requires_snapshot
def test_cyclic_result_is_seed_independent(site, sizing):
    """The fixed point must be a property of the system, not of where we started."""
    solar_mw, battery_mw, duration_h = sizing
    common = dict(solar_mw=solar_mw, battery_mw=battery_mw, battery_duration_h=duration_h)
    runs = [
        simulate_cyclic(site, FixedLoadController(), initial_soc_pct=seed, **common)
        for seed in (0.0, 25.0, UPSTREAM_INITIAL_SOC_PCT, 100.0)
    ]
    starts = [r.metrics["soc_start_mwh"] for r in runs]
    computes = [r.metrics["compute_units"] for r in runs]
    assert max(starts) - min(starts) < 1e-6
    assert max(computes) - min(computes) < 1e-9


@requires_weather
@requires_snapshot
def test_fixed_mode_default_is_unchanged(site, sizing):
    """The Milestone 1 gate depends on this default. Guard it explicitly."""
    solar_mw, battery_mw, duration_h = sizing
    run = simulate(
        site, FixedLoadController(), solar_mw=solar_mw,
        battery_mw=battery_mw, battery_duration_h=duration_h,
    )
    assert run.metadata["initial_soc_pct"] == UPSTREAM_INITIAL_SOC_PCT == 75.0
    assert run.metadata["initial_soc_mode"] == "fixed"
    expected = 0.75 * battery_mw * duration_h
    assert run.metrics["soc_start_mwh"] == pytest.approx(expected)


@requires_weather
@requires_snapshot
def test_cyclic_energy_conservation_still_holds(site, sizing):
    solar_mw, battery_mw, duration_h = sizing
    run = simulate_cyclic(
        site, SimpleThrottleController(), solar_mw=solar_mw,
        battery_mw=battery_mw, battery_duration_h=duration_h,
    )
    soc = run.series("battery_soc_mwh")
    charge = run.series("battery_charge_mw")
    discharge = run.series("battery_discharge_mw")
    capacity = battery_mw * duration_h
    assert soc.min() >= -1e-9
    assert soc.max() <= capacity + 1e-9
    assert charge.max() <= battery_mw + 1e-9
    assert discharge.max() <= battery_mw + 1e-9
    np.testing.assert_allclose(np.diff(soc), (charge - discharge)[:-1], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Discontinuous controllers: the fixed point can fail to exist
# ---------------------------------------------------------------------------

@requires_weather
def test_bang_bang_controller_falls_back_to_bisection(site):
    """A step-function controller can put the fixed-point iteration in a cycle.

    ``SimpleThrottleController`` maps starting SOC to year-end SOC through
    discrete bands, so a hair's change in the start can flip a band and move the
    end by whole MWh. Plain iteration was observed orbiting with period 5 at
    low-solar sizings; bisecting the residual is robust to those jumps.

    This configuration is one of the ones that used to raise.
    """
    run = simulate_cyclic(
        site, SimpleThrottleController(),
        solar_mw=30.0, battery_mw=124.46, battery_duration_h=12.54,
    )
    info = run.metadata["cyclic_soc"]
    assert info["converged"]
    assert info["phase"] != "fixed_point", "expected the fallback to be exercised"
    # The residual must be negligible against annual throughput, not merely small.
    capacity = 124.46 * 12.54
    assert abs(info["final_gap_mwh"]) <= info["relaxed_tolerance_mwh"]
    assert abs(info["final_gap_mwh"]) / capacity < 1e-3
    assert abs(run.metrics["cyclic_soc_residual_mwh"]) == pytest.approx(
        abs(info["final_gap_mwh"])
    )


@requires_weather
@pytest.mark.parametrize(
    "battery_mw,duration_h",
    [(122.47, 14.72), (118.66, 4.53), (34.11, 13.20), (46.89, 10.41)],
)
def test_stressed_sizings_never_raise(site, battery_mw, duration_h):
    """Experiment B's optimiser probes extreme corners; none may crash it."""
    run = simulate_cyclic(
        site, SimpleThrottleController(),
        solar_mw=30.0, battery_mw=battery_mw, battery_duration_h=duration_h,
    )
    assert run.metadata["cyclic_soc"]["converged"]
    assert abs(run.metrics["cyclic_soc_residual_mwh"]) <= (
        run.metadata["cyclic_soc"]["relaxed_tolerance_mwh"]
    )


@requires_weather
@requires_snapshot
def test_smooth_controllers_still_take_the_fast_path(site, sizing):
    """The fallback must not slow down the common case."""
    solar_mw, battery_mw, duration_h = sizing
    run = simulate_cyclic(
        site, FixedLoadController(), solar_mw=solar_mw,
        battery_mw=battery_mw, battery_duration_h=duration_h,
    )
    info = run.metadata["cyclic_soc"]
    assert info["phase"] == "fixed_point"
    assert info["evaluations"] <= 3
