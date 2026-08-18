"""MPC tests.

The planner is a second, independent model of the plant. That is the risk: an
LP can be beautifully optimal against physics that does not match the
simulator, and the result looks credible either way. So the central test here
is not "is the LP optimal" but **"does the LP's own prediction match what the
dispatcher actually measures when the plan is executed"**. Everything else
guards the pieces.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace

from flexcompute.control import Observation
from flexcompute.dispatch import simulate_cyclic
from flexcompute.forecast import PerfectSolarForecast, SolarForecast
from flexcompute.gpu import get_curve
from flexcompute.mpc import (
    AnnualPerfectForesightPlanner,
    PlantModel,
    ScheduleController,
    build_plant_model,
    solve_schedule,
)
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot

from conftest import BASELINE_SCENARIO, requires_weather

SNAPSHOT_PATH = SNAPSHOT_DIR / f"{BASELINE_SCENARIO.label()}.json"

requires_snapshot = pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(), reason="run scripts/run_baseline.py first"
)


# ---------------------------------------------------------------------------
# Synthetic plant: small, hand-checkable
# ---------------------------------------------------------------------------

def _plant(n=24, solar=None, battery_mw=5.0, battery_mwh=20.0, demand=10.0,
           fixed_cooling_bus_mw=0.0) -> PlantModel:
    curve = get_curve()
    hull = curve.concave_hull()
    slopes = np.diff(hull.compute_fraction) / np.diff(hull.power_fraction)
    intercepts = hull.compute_fraction[:-1] - slopes * hull.power_fraction[:-1]
    solar_dc = np.full(n, 40.0) if solar is None else np.asarray(solar, dtype=float)
    return PlantModel(
        alpha=np.full(n, 1.2),
        beta=np.full(n, fixed_cooling_bus_mw),
        solar_dc_mw=solar_dc,
        demand_mw=np.full(n, demand),
        m_solar_bus=1.12,
        m_solar_batt=1.09,
        m_batt_bus=1.19,
        battery_mw=battery_mw,
        battery_mwh=battery_mwh,
        min_power_fraction=hull.power_fraction[0],
        hull_slopes=slopes,
        hull_intercepts=intercepts,
    )


def test_abundant_solar_runs_at_full_demand():
    """Nothing scarce: the optimum is simply to run flat out every hour."""
    solution = solve_schedule(_plant(), initial_soc_mwh=0.0)
    assert solution.status == "optimal"
    np.testing.assert_allclose(solution.it_power_mw, 10.0, rtol=1e-6)
    assert solution.compute_units.sum() == pytest.approx(24.0, rel=1e-6)


def test_scarcity_forces_throttling_not_failure():
    """With too little energy the plan throttles; it never plans a shortfall."""
    solar = np.concatenate([np.full(6, 30.0), np.zeros(18)])
    solution = solve_schedule(_plant(solar=solar), initial_soc_mwh=10.0)
    assert solution.status == "optimal"
    power_fraction = solution.it_power_mw / 10.0
    assert power_fraction.min() < 0.99            # it did throttle
    assert power_fraction.min() >= _plant().min_power_fraction - 1e-9
    assert solution.compute_units.sum() < 24.0


def test_never_charges_and_discharges_in_the_same_hour():
    """Round-trip losses make it strictly wasteful, so no binary is needed.

    If this ever fails the LP has found a degenerate optimum and the
    formulation needs a complementarity constraint.
    """
    rng = np.random.default_rng(3)
    for _ in range(5):
        solar = np.clip(rng.normal(20, 25, 48), 0, None)
        solution = solve_schedule(_plant(n=48, solar=solar), initial_soc_mwh=8.0)
        both = (solution.charge_mw > 1e-6) & (solution.discharge_mw > 1e-6)
        assert not both.any()


def test_plan_respects_battery_bounds_and_ratings():
    rng = np.random.default_rng(11)
    solar = np.clip(rng.normal(18, 22, 72), 0, None)
    plant = _plant(n=72, solar=solar, battery_mw=4.0, battery_mwh=25.0)
    solution = solve_schedule(plant, initial_soc_mwh=12.0)
    assert solution.soc_mwh.min() >= -1e-6
    assert solution.soc_mwh.max() <= plant.battery_mwh + 1e-6
    assert solution.charge_mw.max() <= plant.battery_mw + 1e-6
    assert solution.discharge_mw.max() <= plant.battery_mw + 1e-6
    np.testing.assert_allclose(
        np.diff(solution.soc_mwh), solution.charge_mw - solution.discharge_mw, atol=1e-6
    )


def test_cyclic_constraint_forbids_net_drain():
    solar = np.concatenate([np.full(4, 25.0), np.zeros(20)])
    solution = solve_schedule(_plant(solar=solar), initial_soc_mwh=None, cyclic=True)
    assert solution.soc_mwh[-1] >= solution.soc_mwh[0] - 1e-6


def test_free_initial_soc_beats_a_fixed_empty_start():
    """Sanity on the degrees of freedom: more freedom cannot hurt the objective."""
    solar = np.concatenate([np.full(6, 22.0), np.zeros(18)])
    plant = _plant(solar=solar)
    fixed = solve_schedule(plant, initial_soc_mwh=0.0)
    free = solve_schedule(plant, initial_soc_mwh=None, cyclic=True)
    assert free.objective >= fixed.objective - 1e-6


def test_terminal_value_discourages_draining():
    """Priced terminal energy must leave more in the battery, not less."""
    solar = np.concatenate([np.full(6, 25.0), np.zeros(18)])
    plant = _plant(solar=solar)
    without = solve_schedule(plant, initial_soc_mwh=15.0, terminal_value_per_mwh=0.0)
    priced = solve_schedule(plant, initial_soc_mwh=15.0, terminal_value_per_mwh=0.5)
    assert priced.soc_mwh[-1] >= without.soc_mwh[-1] - 1e-6


def test_marginal_compute_per_mwh_is_positive_and_finite():
    value = _plant().marginal_compute_per_battery_mwh()
    assert 0.0 < value < 1.0


def test_inverter_cap_follows_the_believed_solar_series():
    """A forecast that differs from the truth must be capped on its own terms.

    Carrying the cap as a ratio against the realised series was correct only
    because perfect foresight makes belief and truth identical; it would have
    silently mis-capped every noisy forecast.
    """
    plant = replace(_plant(n=4, solar=np.array([10.0, 20.0, 40.0, 80.0])),
                    inverter_cap_mw_dc=30.0)
    np.testing.assert_allclose(
        plant.solar_to_bus_max, np.array([10.0, 20.0, 30.0, 30.0]) / 1.12
    )
    believed = plant.with_solar(np.array([80.0, 40.0, 20.0, 10.0]))
    np.testing.assert_allclose(
        believed.solar_to_bus_max, np.array([30.0, 30.0, 20.0, 10.0]) / 1.12
    )


def test_with_solar_rejects_a_mismatched_length():
    with pytest.raises(ValueError):
        _plant(n=4).with_solar(np.zeros(3))


def test_window_truncates_past_the_end():
    plant = _plant(n=10)
    window = plant.window(8, 6)
    assert window.horizon == 2
    assert len(window.solar_dc_mw) == 2
    assert len(window.demand_mw) == 2
    assert len(window.beta) == 2


def test_fixed_cooling_raises_the_floor_on_bus_load():
    """With a beta term the plant draws power even at minimum GPU power.

    Mirrors the dispatcher's affine bus load (ASSUMPTIONS Q3); if the planner
    kept the proportional form it would plan for a plant that does not exist.
    """
    plain = solve_schedule(_plant(), initial_soc_mwh=0.0)
    with_fixed = solve_schedule(
        _plant(fixed_cooling_bus_mw=2.0), initial_soc_mwh=0.0
    )
    # Abundant solar in both, so compute is unaffected ...
    assert with_fixed.compute_units.sum() == pytest.approx(
        plain.compute_units.sum(), rel=1e-6
    )
    # ... but the fixed draw has to come from somewhere.
    assert with_fixed.it_power_mw.sum() == pytest.approx(plain.it_power_mw.sum(), rel=1e-6)
    assert np.all(with_fixed.unserved_bus_mwh <= 1e-9)


# ---------------------------------------------------------------------------
# Forecast interface
# ---------------------------------------------------------------------------

def test_perfect_forecast_returns_the_truth():
    truth = np.arange(100, dtype=float)
    forecast = PerfectSolarForecast(truth)
    assert isinstance(forecast, SolarForecast)
    np.testing.assert_array_equal(forecast.horizon(10, 5), truth[10:15])


def test_perfect_forecast_truncates_and_never_wraps():
    """December must not see January (acausal), nor a fake week of night.

    Zero-padding the tail was observed driving the LP infeasible at long
    horizons -- an artefact of the padding, not of the plant.
    """
    truth = np.arange(10, dtype=float) + 1.0
    window = PerfectSolarForecast(truth).horizon(8, 5)
    assert window.tolist() == [9.0, 10.0]        # truncated, not padded
    assert len(window) == 2


def test_perfect_forecast_declares_itself_an_upper_bound():
    meta = PerfectSolarForecast(np.zeros(10)).metadata()
    assert meta["kind"] == "perfect_foresight"
    assert "not achievable" in meta["caveat"]


# ---------------------------------------------------------------------------
# Belief vs truth: the substitution that Milestone 6 rests on
# ---------------------------------------------------------------------------

def _observation(t: int, soc_mwh: float) -> Observation:
    return Observation(
        t=t, hour_of_day=t % 24, soc_mwh=soc_mwh, soc_fraction=soc_mwh / 20.0,
        battery_energy_mwh=20.0, battery_power_mw=5.0, solar_dc_mw=0.0,
        pue=1.1, nameplate_it_mw=10.0, unconstrained_demand_mw=10.0,
        min_operating_it_mw=1.0,
    )


def _mpc_fixtures(sigma_24h: float, n: int = 240, seed: int = 3):
    """A plant with real day/night structure, plus a belief about it."""
    from flexcompute.forecast import NoisySolarForecast
    from flexcompute.gpu import GpuFleet

    hours = np.arange(n)
    day = np.clip(np.sin((hours % 24 - 6) / 12 * np.pi), 0.0, None)
    cloud = np.repeat(np.random.default_rng(seed).uniform(0.1, 1.0, n // 24), 24)
    solar = 30.0 * day * cloud

    plant = _plant(n=n, solar=solar, battery_mw=5.0, battery_mwh=20.0)
    fleet = GpuFleet(curve=get_curve(), total_gpus=10_000)
    return plant, fleet, NoisySolarForecast(solar, sigma_24h=sigma_24h, seed=seed)


def test_a_perfect_belief_reproduces_the_perfect_foresight_controller():
    """The two controllers must be the same machine, differing only in belief.

    This is what licenses attributing the perfect-vs-noisy gap to *information*
    rather than to some incidental difference in how the two are wired. Checked
    as bit-identical actions, not as a tolerance.
    """
    from flexcompute.mpc import ForecastMPCController, PerfectForesightMPCController

    plant, fleet, _ = _mpc_fixtures(sigma_24h=0.0)
    truth = PerfectSolarForecast(plant.solar_dc_mw)
    common = dict(model=plant, fleet=fleet, horizon_hours=48)

    general = ForecastMPCController(forecast=truth, **common)
    perfect = PerfectForesightMPCController(forecast=truth, **common)
    general.reset(horizon=plant.horizon)
    perfect.reset(horizon=plant.horizon)

    for t, soc in ((0, 10.0), (37, 4.0), (100, 18.0), (200, 0.0)):
        obs = _observation(t, soc)
        assert general.choose_power(obs) == perfect.choose_power(obs)


def test_a_zero_error_noisy_belief_is_also_indistinguishable():
    """sigma=0 must reach the LP as the truth, not merely as something close."""
    from flexcompute.mpc import ForecastMPCController

    plant, fleet, quiet = _mpc_fixtures(sigma_24h=0.0)
    common = dict(model=plant, fleet=fleet, horizon_hours=48)
    noisy = ForecastMPCController(forecast=quiet, **common)
    exact = ForecastMPCController(
        forecast=PerfectSolarForecast(plant.solar_dc_mw), **common
    )
    for t, soc in ((12, 8.0), (90, 3.0)):
        assert noisy.choose_power(_observation(t, soc)) == exact.choose_power(
            _observation(t, soc)
        )


def test_a_wrong_belief_actually_changes_what_the_controller_does():
    """The belief must reach the solver -- otherwise the experiment is a no-op.

    A controller that quietly ignored its forecast would produce a reassuring
    "forecast error costs nothing" result. It must be shown to bite.
    """
    from flexcompute.mpc import ForecastMPCController

    plant, fleet, wrong = _mpc_fixtures(sigma_24h=0.30)
    common = dict(model=plant, fleet=fleet, horizon_hours=48)
    believer = ForecastMPCController(forecast=wrong, **common)
    knower = ForecastMPCController(
        forecast=PerfectSolarForecast(plant.solar_dc_mw), **common
    )

    # Scarce battery, so the plan depends on what it thinks tomorrow brings.
    differences = sum(
        believer.choose_power(_observation(t, 4.0))
        != knower.choose_power(_observation(t, 4.0))
        for t in range(0, 200, 7)
    )
    assert differences > 0


def test_perfect_foresight_controller_refuses_a_forecast_that_can_be_wrong():
    """Design rule 4, made mechanical rather than conventional.

    A run labelled ``perfect_foresight_mpc`` cannot have been handed a belief,
    and a noisy run cannot inherit the perfect-foresight label by accident.
    """
    from flexcompute.mpc import PerfectForesightMPCController

    plant, fleet, noisy = _mpc_fixtures(sigma_24h=0.15)
    with pytest.raises(TypeError, match="PerfectSolarForecast"):
        PerfectForesightMPCController(model=plant, fleet=fleet, forecast=noisy)


def test_forecast_mpc_metadata_carries_the_belief_it_used():
    from flexcompute.mpc import ForecastMPCController

    plant, fleet, noisy = _mpc_fixtures(sigma_24h=0.15)
    meta = ForecastMPCController(model=plant, fleet=fleet, forecast=noisy).metadata()
    assert meta["name"] == "forecast_mpc"
    assert meta["forecast"]["kind"] == "noisy"
    assert meta["forecast"]["parameters"]["sigma_24h"] == 0.15


# ---------------------------------------------------------------------------
# Schedule replay
# ---------------------------------------------------------------------------

def test_schedule_controller_replays_exactly():
    schedule = np.linspace(1.0, 9.0, 8760)
    controller = ScheduleController(schedule)
    controller.reset(horizon=8760)
    for t in (0, 100, 8759):
        obs = Observation(
            t=t, hour_of_day=t % 24, soc_mwh=1.0, soc_fraction=0.5,
            battery_energy_mwh=2.0, battery_power_mw=1.0, solar_dc_mw=0.0,
            pue=1.1, nameplate_it_mw=9.0, unconstrained_demand_mw=9.0,
            min_operating_it_mw=0.9,
        )
        assert controller.choose_power(obs) == schedule[t]


def test_schedule_controller_rejects_a_short_schedule():
    with pytest.raises(ValueError):
        ScheduleController(np.zeros(10)).reset(horizon=8760)


# ---------------------------------------------------------------------------
# The one that matters: planner model == simulator
# ---------------------------------------------------------------------------

@requires_weather
@requires_snapshot
def test_annual_plan_prediction_matches_the_dispatcher(site):
    """The LP and the simulator must agree about the same schedule.

    Any gap means the planner is optimising against physics the simulator does
    not implement, which would silently invalidate every MPC result.
    """
    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    model, _ = build_plant_model(
        site, solar_mw=s["solar_mw_dc"], battery_mw=s["battery_mw"],
        battery_duration_h=s["battery_duration_h"],
    )
    solution = AnnualPerfectForesightPlanner(model).solve()

    from flexcompute.dispatch import simulate

    run = simulate(
        site, ScheduleController(solution.it_power_mw),
        solar_mw=s["solar_mw_dc"], battery_mw=s["battery_mw"],
        battery_duration_h=s["battery_duration_h"],
        initial_soc_pct=100.0 * solution.soc_mwh[0] / model.battery_mwh,
    )
    assert run.metrics["compute_units"] == pytest.approx(
        float(solution.compute_units.sum()), abs=1e-6
    )
    # A feasible plan executed under perfect foresight must never fall short.
    assert run.metrics["involuntary_shortfall_mwh"] == pytest.approx(0.0, abs=1e-6)


@requires_weather
@requires_snapshot
def test_annual_plan_is_cyclic_and_takes_no_free_energy(site):
    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    model, _ = build_plant_model(
        site, solar_mw=s["solar_mw_dc"], battery_mw=s["battery_mw"],
        battery_duration_h=s["battery_duration_h"],
    )
    solution = AnnualPerfectForesightPlanner(model).solve()
    assert solution.soc_mwh[-1] >= solution.soc_mwh[0] - 1e-6


@requires_weather
@requires_snapshot
def test_perfect_foresight_is_an_upper_bound_on_the_heuristics(site):
    """The ceiling must actually be a ceiling.

    Run at a de-rated sizing where stored energy binds -- at the fixed-load
    optimum almost nothing binds and every policy scores within a rounding
    error of every other.
    """
    from flexcompute.experiments import Sizing, run_strategy

    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    sizing = Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"]).scaled(0.4)

    ceiling = run_strategy(site, "perfect_foresight_annual", sizing).metrics["compute_units"]
    for strategy in ("fixed_load", "simple_throttle"):
        assert run_strategy(site, strategy, sizing).metrics["compute_units"] <= ceiling + 1e-6


# ---------------------------------------------------------------------------
# The compiled window LP is a speed optimisation and nothing else
# ---------------------------------------------------------------------------
#
# ForecastMPCController re-solves a parametrised, pre-canonicalised LP instead
# of rebuilding the problem every hour. That is worth ~2.7x on a simulated
# year, which is what makes the multi-year sizing search affordable -- but a
# performance path that quietly changes the answer would invalidate every MPC
# number in the project. These tests are the gate on it.

def test_compiled_window_lp_is_dpp_compliant():
    """If it is not DPP, cvxpy re-canonicalises every solve and caches nothing.

    The speedup would silently vanish while the tests still passed, so the
    property is asserted rather than assumed. ``_CompiledWindowLP`` also raises
    on construction if this fails; this pins it at the level of the real model.
    """
    from flexcompute.mpc import _CompiledWindowLP

    model = _plant(n=48)
    compiled = _CompiledWindowLP(model, solver="HIGHS")
    assert compiled.problem.is_dcp(dpp=True)


def test_compiled_and_reference_paths_agree_on_a_synthetic_window():
    """Same window, both formulations, same optimum."""
    from flexcompute.mpc import _CompiledWindowLP

    solar = np.concatenate([np.full(8, 30.0), np.zeros(16), np.full(6, 12.0), np.zeros(18)])
    model = _plant(n=48, solar=solar)

    reference = solve_schedule(
        model, initial_soc_mwh=8.0, terminal_value_per_mwh=0.25
    )
    fast = _CompiledWindowLP(model, solver="HIGHS").solve(
        model, initial_soc_mwh=8.0, terminal_value_per_mwh=0.25
    )

    np.testing.assert_allclose(fast.it_power_mw, reference.it_power_mw, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(fast.soc_mwh, reference.soc_mwh, rtol=1e-8, atol=1e-8)
    assert fast.compute_units.sum() == pytest.approx(reference.compute_units.sum(), rel=1e-9)


def test_compiled_path_prices_shortfall_the_same_way():
    """The unserved penalty is recomputed per window; check it transfers.

    Priced off the window's own mean bus load in both paths. A mismatch would
    only show up in energy-starved windows, so this uses one.
    """
    from flexcompute.mpc import _CompiledWindowLP

    model = _plant(n=24, solar=np.zeros(24), battery_mw=1.0, battery_mwh=2.0)
    reference = solve_schedule(model, initial_soc_mwh=0.5, terminal_value_per_mwh=0.0)
    fast = _CompiledWindowLP(model, solver="HIGHS").solve(
        model, initial_soc_mwh=0.5, terminal_value_per_mwh=0.0
    )
    np.testing.assert_allclose(
        fast.unserved_bus_mwh, reference.unserved_bus_mwh, rtol=1e-8, atol=1e-8
    )


@requires_weather
@requires_snapshot
def test_compiled_mpc_reproduces_the_reference_path_over_a_full_year(site):
    """The one that matters: a whole simulated year, both paths, same answer.

    Not asserted bit-identical -- HiGHS canonicalises the two formulations in a
    different order, so the last couple of ulps differ. Asserted instead at a
    tolerance thousands of times tighter than anything that could change a
    conclusion: hourly actions to 1e-9 relative, annual compute to 1e-12.
    """
    from flexcompute.dispatch import simulate
    from flexcompute.experiments import Sizing
    from flexcompute.mpc import PerfectForesightMPCController

    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    sizing = Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"]).scaled(0.4)
    kwargs = dict(
        solar_mw=sizing.solar_mw,
        battery_mw=sizing.battery_mw,
        battery_duration_h=sizing.duration_h,
    )

    def run(compiled: bool):
        model, fleet = build_plant_model(site, **kwargs)
        controller = PerfectForesightMPCController(
            model=model,
            fleet=fleet,
            forecast=PerfectSolarForecast(model.solar_dc_mw),
            horizon_hours=48,
            compile_window=compiled,
        )
        return simulate(site, controller, **kwargs)

    reference, fast = run(False), run(True)

    np.testing.assert_allclose(
        fast.series("controller_target_it_mw"),
        reference.series("controller_target_it_mw"),
        rtol=1e-9,
        atol=1e-9,
    )
    for key in ("compute_units", "involuntary_shortfall_mwh", "voluntary_throttle_mwh"):
        assert fast.metrics[key] == pytest.approx(reference.metrics[key], rel=1e-12, abs=1e-9)


def test_compiled_path_can_be_switched_off_and_is_recorded():
    """A result must say which path produced it."""
    from flexcompute.mpc import ForecastMPCController

    from flexcompute.gpu import GpuFleet

    model = _plant(n=48)
    fleet = GpuFleet(curve=get_curve(), total_gpus=10)
    controller = ForecastMPCController(
        model=model, fleet=fleet,
        forecast=PerfectSolarForecast(model.solar_dc_mw), compile_window=False,
    )
    assert controller.metadata()["parameters"]["compiled_window_lp"] is False
