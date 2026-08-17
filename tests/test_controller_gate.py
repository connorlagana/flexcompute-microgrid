"""The Milestone 2 gate.

Introducing a controller must change nothing about the fixed-load result. Three
claims, all enforced here:

1. ``FixedLoadController`` sets ``controller_target == unconstrained_demand``
   at every one of the 8760 timesteps, exactly.
2. Routing it through the closed-loop simulator reproduces upstream's dispatch
   **bit-for-bit** -- not "to within a tolerance".
3. The four-way power decomposition is internally consistent for every
   controller, and voluntary throttling is never confused with shortfall.

If any of these fail, the seam is wrong and no result computed on top of it
means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.baseline import evaluate_fixed_sizing
from flexcompute.control import (
    ComputeController,
    FixedLoadController,
    Observation,
    SimpleThrottleController,
)
from flexcompute.dispatch import simulate
from flexcompute.gpu import get_curve
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot

from conftest import BASELINE_SCENARIO, requires_weather

SNAPSHOT_PATH = SNAPSHOT_DIR / f"{BASELINE_SCENARIO.label()}.json"

requires_snapshot = pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(),
    reason=f"No stored baseline at {SNAPSHOT_PATH}. Run scripts/run_baseline.py first.",
)


@pytest.fixture(scope="module")
def sizing():
    if not SNAPSHOT_PATH.exists():
        pytest.skip("no baseline snapshot")
    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    return s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"]


@pytest.fixture(scope="module")
def fixed_run(site, sizing):
    solar_mw, battery_mw, duration_h = sizing
    return simulate(
        site, FixedLoadController(), solar_mw=solar_mw,
        battery_mw=battery_mw, battery_duration_h=duration_h,
    )


@pytest.fixture(scope="module")
def throttle_run(site, sizing):
    solar_mw, battery_mw, duration_h = sizing
    return simulate(
        site, SimpleThrottleController(), solar_mw=solar_mw,
        battery_mw=battery_mw, battery_duration_h=duration_h,
    )


# ---------------------------------------------------------------------------
# Gate 1: the controller asks for exactly what the workload wants
# ---------------------------------------------------------------------------

@requires_weather
@requires_snapshot
def test_fixed_controller_target_equals_unconstrained_demand(fixed_run):
    target = fixed_run.series("controller_target_it_mw")
    demand = fixed_run.series("unconstrained_demand_mw")
    assert np.array_equal(target, demand)


@requires_weather
@requires_snapshot
def test_fixed_controller_never_throttles_voluntarily(fixed_run):
    assert np.array_equal(fixed_run.series("voluntary_throttle_mw"), np.zeros(8760))
    assert fixed_run.metrics["voluntary_throttle_mwh"] == 0.0
    assert fixed_run.metrics["hours_throttled"] == 0
    assert fixed_run.metrics["hours_parked"] == 0


# ---------------------------------------------------------------------------
# Gate 2: bit-for-bit reproduction of upstream dispatch
# ---------------------------------------------------------------------------

@requires_weather
@requires_snapshot
@pytest.mark.parametrize(
    "upstream_col,ours",
    [
        ("solar_dc_mw", "solar_dc_mw"),
        ("solar_at_load_mw", "solar_at_bus_mw"),
        ("it_load_mw", "controller_target_it_mw"),
        ("cooling_load_mw", "cooling_load_mw"),
        ("battery_soc_mwh", "battery_soc_mwh"),
        ("battery_charge_mw", "battery_charge_mw"),
        ("battery_discharge_mw", "battery_discharge_mw"),
        ("curtailed_solar_mw", "curtailed_solar_mw"),
        ("unmet_load_mw", "unmet_load_mw"),
    ],
)
def test_closed_loop_matches_upstream_bit_for_bit(site, sizing, fixed_run, upstream_col, ours):
    solar_mw, battery_mw, duration_h = sizing
    reference = evaluate_fixed_sizing(
        site, solar_mw, battery_mw, battery_duration_h=duration_h
    )
    expected = reference.sim.hourly_data[upstream_col].to_numpy(dtype=float)
    actual = fixed_run.series(ours)
    assert np.array_equal(expected, actual), (
        f"{ours} diverged from upstream {upstream_col}; "
        f"max |diff| = {np.abs(expected - actual).max():.3e}"
    )


@requires_weather
@requires_snapshot
@pytest.mark.parametrize(
    "key",
    ["uptime_pct", "solar_generation_mwh", "solar_curtailed_mwh",
     "battery_charged_mwh", "battery_discharged_mwh", "battery_cycles_per_year"],
)
def test_headline_metrics_match_upstream_exactly(site, sizing, fixed_run, key):
    solar_mw, battery_mw, duration_h = sizing
    reference = evaluate_fixed_sizing(
        site, solar_mw, battery_mw, battery_duration_h=duration_h
    )
    assert fixed_run.metrics[key] == reference.metrics[key]


@requires_weather
@requires_snapshot
def test_fixed_run_reproduces_the_committed_snapshot(fixed_run):
    """Tie the closed loop back to the Milestone 1 contract itself."""
    stored = load_snapshot(SNAPSHOT_PATH)["optimized"]["year_0"]
    for key in ("uptime_pct", "solar_generation_mwh", "solar_curtailed_mwh",
                "battery_discharged_mwh", "battery_cycles_per_year"):
        assert fixed_run.metrics[key] == pytest.approx(stored[key], rel=1e-12), key


# ---------------------------------------------------------------------------
# Gate 3: the four-way decomposition is coherent
# ---------------------------------------------------------------------------

@requires_weather
@requires_snapshot
@pytest.mark.parametrize("run_name", ["fixed", "throttle"])
def test_four_way_power_decomposition(request, run_name):
    run = request.getfixturevalue(f"{run_name}_run")
    demand = run.series("unconstrained_demand_mw")
    target = run.series("controller_target_it_mw")
    delivered = run.series("delivered_it_mw")
    voluntary = run.series("voluntary_throttle_mw")
    involuntary = run.series("involuntary_shortfall_mw")

    # The two gaps reconstruct the whole
    np.testing.assert_allclose(voluntary, demand - target, rtol=0, atol=0)
    np.testing.assert_allclose(involuntary, target - delivered, rtol=0, atol=0)
    np.testing.assert_allclose(demand - delivered, voluntary + involuntary, rtol=1e-12, atol=1e-12)

    # Ordering: you cannot deliver more than was targeted, or target more than wanted
    assert np.all(target <= demand + 1e-12)
    assert np.all(delivered <= target + 1e-12)
    assert np.all(delivered >= -1e-12)

    # Neither gap is ever negative: a "shortfall" that is really a surplus
    # would silently flatter the controller.
    assert voluntary.min() >= -1e-12
    assert involuntary.min() >= -1e-12


@requires_weather
@requires_snapshot
def test_shortfall_only_occurs_when_the_bus_is_short(throttle_run):
    """Involuntary shortfall must be caused by unmet bus load, never by the
    controller's own choice."""
    involuntary = throttle_run.series("involuntary_shortfall_mw")
    unmet = throttle_run.series("unmet_load_mw")
    assert np.all((involuntary <= 1e-12) | (unmet > 0.0))


# ---------------------------------------------------------------------------
# The throttle controller actually does something
# ---------------------------------------------------------------------------

@requires_weather
@requires_snapshot
def test_throttle_controller_reacts_to_stored_energy(fixed_run, throttle_run):
    """The point of Milestone 2: demand responds to the battery.

    Not a claim that it responds *well* -- SimpleThrottleController is
    deliberately naive. Only that the loop is closed.
    """
    assert throttle_run.metrics["hours_throttled"] > 0
    assert throttle_run.metrics["voluntary_throttle_mwh"] > 0
    # It conserves stored energy relative to the fixed load ...
    assert throttle_run.metrics["soc_min_mwh"] > fixed_run.metrics["soc_min_mwh"]
    # ... and it converts that into avoided failures.
    assert throttle_run.metrics["involuntary_shortfall_mwh"] < fixed_run.metrics["involuntary_shortfall_mwh"]


@requires_weather
@requires_snapshot
def test_energy_is_conserved_under_both_controllers(fixed_run, throttle_run):
    """Same physical laws, whoever is driving."""
    for run in (fixed_run, throttle_run):
        soc = run.series("battery_soc_mwh")
        charge = run.series("battery_charge_mw")
        discharge = run.series("battery_discharge_mw")
        capacity = run.metadata["sizing"]["battery_mwh"]
        power = run.metadata["sizing"]["battery_mw"]

        assert soc.min() >= -1e-9
        assert soc.max() <= capacity + 1e-9
        assert charge.max() <= power + 1e-9
        assert discharge.max() <= power + 1e-9
        assert not np.any((charge > 1e-12) & (discharge > 1e-12))
        np.testing.assert_allclose(np.diff(soc), (charge - discharge)[:-1], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Causality: a controller cannot see the future
# ---------------------------------------------------------------------------

def test_observation_exposes_no_future_information():
    """Structural guarantee, checked as a test so it cannot rot.

    Perfect foresight has to arrive as an explicit SolarForecast dependency
    (Milestone 4), never as a field that happens to be lying around.
    """
    fields = set(Observation.__dataclass_fields__)
    forbidden = {"solar_forecast", "future_solar_mw", "solar_dc_mw_series",
                 "site", "horizon_solar", "tomorrow"}
    assert not (fields & forbidden)
    # Every field is a scalar snapshot of "now" -- no arrays, no site handle.
    for name, field in Observation.__dataclass_fields__.items():
        assert field.type in {"int", "float"}, f"{name} is {field.type}, not a scalar"


def test_controllers_satisfy_the_protocol():
    for controller in (FixedLoadController(), SimpleThrottleController()):
        assert isinstance(controller, ComputeController)
        meta = controller.metadata()
        assert set(meta) >= {"name", "kind", "parameters"}


def test_controllers_are_deterministic():
    obs = Observation(
        t=100, hour_of_day=4, soc_mwh=200.0, soc_fraction=0.4,
        battery_energy_mwh=500.0, battery_power_mw=120.0, solar_dc_mw=0.0,
        pue=1.13, nameplate_it_mw=9.1, unconstrained_demand_mw=9.1,
        min_operating_it_mw=2.8,
    )
    for controller in (FixedLoadController(), SimpleThrottleController()):
        values = {controller.choose_power(obs) for _ in range(10)}
        assert len(values) == 1


def test_simple_throttle_bands_are_monotone_in_soc():
    """More stored energy must never mean less requested power."""
    controller = SimpleThrottleController()
    previous = -1.0
    for soc_fraction in np.linspace(0.0, 1.0, 101):
        obs = Observation(
            t=0, hour_of_day=0, soc_mwh=soc_fraction * 500.0,
            soc_fraction=float(soc_fraction), battery_energy_mwh=500.0,
            battery_power_mw=120.0, solar_dc_mw=0.0, pue=1.13,
            nameplate_it_mw=9.1, unconstrained_demand_mw=9.1,
            min_operating_it_mw=2.8,
        )
        power = controller.choose_power(obs)
        assert power >= previous - 1e-12
        previous = power


def test_simple_throttle_rejects_unordered_bands():
    with pytest.raises(ValueError):
        SimpleThrottleController(bands=((0.2, 0.5), (0.6, 1.0)))
