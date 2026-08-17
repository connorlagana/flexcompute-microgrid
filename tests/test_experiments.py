"""Experiment harness tests.

The harness exists to guarantee comparability. These tests check that guarantee
directly: every strategy must see the same world, and the strategies must order
themselves the way physics requires (nothing beats perfect foresight).

Marked slow where the MPC is involved -- a receding-horizon year is ~200 s
because the cyclic fixed point runs it twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.experiments import (
    KNOWN_STRATEGIES,
    STRATEGY_ORDER,
    Sizing,
    default_forecast_factory,
    infrastructure_for_target_compute,
    run_experiment_a,
    run_strategy,
)

from conftest import BASELINE_SCENARIO, requires_weather
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot

SNAPSHOT_PATH = SNAPSHOT_DIR / f"{BASELINE_SCENARIO.label()}.json"

requires_snapshot = pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(), reason="run scripts/run_baseline.py first"
)

FAST_STRATEGIES = ("fixed_load", "simple_throttle", "perfect_foresight_annual")


@pytest.fixture(scope="module")
def base_sizing():
    if not SNAPSHOT_PATH.exists():
        pytest.skip("no baseline snapshot")
    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    return Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"])


def test_sizing_scales_both_axes_together():
    sizing = Sizing(100.0, 50.0, 4.0)
    assert sizing.battery_mwh == 200.0
    half = sizing.scaled(0.5)
    assert (half.solar_mw, half.battery_mw, half.battery_mwh) == (50.0, 25.0, 100.0)
    assert half.duration_h == 4.0


@requires_weather
@requires_snapshot
def test_unknown_strategy_is_rejected(site, base_sizing):
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_strategy(site, "telepathy", base_sizing)


@requires_weather
@requires_snapshot
def test_every_strategy_sees_the_same_world(site, base_sizing):
    """The apples-to-apples guarantee, checked rather than assumed."""
    runs = run_experiment_a(site, base_sizing, strategies=FAST_STRATEGIES)
    reference = runs["fixed_load"]

    for name, run in runs.items():
        np.testing.assert_array_equal(
            run.series("solar_dc_mw"), reference.series("solar_dc_mw")
        ), name
        np.testing.assert_array_equal(
            run.series("unconstrained_demand_mw"),
            reference.series("unconstrained_demand_mw"),
        ), name
        np.testing.assert_array_equal(run.series("pue"), reference.series("pue")), name
        assert run.metadata["sizing"] == reference.metadata["sizing"], name


@requires_weather
@requires_snapshot
def test_every_strategy_runs_a_self_sustaining_year(site, base_sizing):
    runs = run_experiment_a(site, base_sizing, strategies=FAST_STRATEGIES)
    for name, run in runs.items():
        assert run.metadata["initial_soc_mode"] == "cyclic", name
        assert run.metrics["soc_net_change_mwh"] == pytest.approx(0.0, abs=1e-6), name


@requires_weather
@requires_snapshot
def test_results_carry_the_gpu_curve_provenance(site, base_sizing):
    """A result must never be quotable without knowing which curve produced it."""
    runs = run_experiment_a(site, base_sizing, strategies=("fixed_load",))
    curve = runs["fixed_load"].metadata["gpu"]["curve"]
    assert curve["kind"] in {"synthetic", "literature_derived", "measured"}
    assert curve["name"]
    assert runs["fixed_load"].metadata["gpu"]["aggregation"] in {"time_shared", "per_device"}


@requires_weather
@requires_snapshot
@pytest.mark.parametrize("scale", [1.0, 0.4])
def test_nothing_beats_perfect_foresight_without_browning_out(site, base_sizing, scale):
    """The ceiling is a ceiling -- among strategies that actually deliver.

    Stated carefully, because it is possible to exceed the annual planner's
    compute by *accepting brownouts*: with fleet-level aggregation a shortfall
    hour still books partial compute. Measured at scale 0.25 the receding-horizon
    MPC does exactly that, beating the ceiling by 0.86 compute-units while
    incurring 7.3 MWh of involuntary shortfall. That is not a better policy; it
    is a different reliability trade, and comparing the compute figures alone
    would hide it.
    """
    sizing = base_sizing.scaled(scale)
    ceiling_run = run_strategy(site, "perfect_foresight_annual", sizing)
    ceiling = ceiling_run.metrics["compute_units"]
    assert ceiling_run.metrics["involuntary_shortfall_mwh"] == pytest.approx(0.0, abs=1e-6)

    for strategy in ("fixed_load", "simple_throttle"):
        run = run_strategy(site, strategy, sizing)
        if run.metrics["involuntary_shortfall_mwh"] > 1e-6:
            continue  # bought compute with brownouts; not a like-for-like comparison
        assert run.metrics["compute_units"] <= ceiling + 1e-6, (
            f"{strategy} beat the shortfall-free ceiling at scale {scale}"
        )


@requires_weather
@requires_snapshot
def test_exceeding_the_ceiling_always_costs_reliability(site, base_sizing):
    """Whenever a strategy out-scores the annual planner, it must have browned out.

    This is the invariant that keeps the compute metric honest. If it ever fails
    with zero shortfall on both sides, the annual LP is not actually finding the
    optimum and every ceiling claim in the project is suspect.
    """
    sizing = base_sizing.scaled(0.25)
    ceiling_run = run_strategy(site, "perfect_foresight_annual", sizing)
    assert ceiling_run.metrics["involuntary_shortfall_mwh"] == pytest.approx(0.0, abs=1e-6)

    for strategy in ("fixed_load", "simple_throttle"):
        run = run_strategy(site, strategy, sizing)
        if run.metrics["compute_units"] > ceiling_run.metrics["compute_units"] + 1e-6:
            assert run.metrics["involuntary_shortfall_mwh"] > 1e-6, (
                f"{strategy} beat the ceiling with no shortfall to pay for it"
            )


@requires_weather
@requires_snapshot
def test_flexibility_matters_more_as_infrastructure_shrinks(site, base_sizing):
    """The core claim of the project, in its weakest testable form.

    The advantage of perfect foresight over a fixed load must grow as the plant
    is de-rated. If it did not, the premise -- that flexible demand substitutes
    for capital -- would be wrong.
    """
    def advantage(scale: float) -> float:
        sizing = base_sizing.scaled(scale)
        fixed = run_strategy(site, "fixed_load", sizing).metrics["compute_units"]
        ceiling = run_strategy(site, "perfect_foresight_annual", sizing).metrics["compute_units"]
        return ceiling / fixed - 1.0

    assert advantage(0.4) > advantage(1.0)
    assert advantage(0.4) > 0.0


def test_infrastructure_interpolation_reads_the_sweep_correctly():
    class FakeRun:
        def __init__(self, compute):
            self.metrics = {"compute_units": compute}

    sweep = {
        0.4: {"s": FakeRun(80.0)},
        0.6: {"s": FakeRun(90.0)},
        0.8: {"s": FakeRun(95.0)},
    }
    # Exactly on a sample
    assert infrastructure_for_target_compute(sweep, "s", 90.0) == pytest.approx(0.6)
    # Halfway between two samples
    assert infrastructure_for_target_compute(sweep, "s", 85.0) == pytest.approx(0.5)
    # Beyond the sweep's reach
    assert infrastructure_for_target_compute(sweep, "s", 999.0) is None


def test_strategy_order_starts_with_the_baseline_and_ends_with_the_ceiling():
    assert STRATEGY_ORDER[0] == "fixed_load"
    assert STRATEGY_ORDER[-1] == "perfect_foresight_annual"


def test_the_forecast_aware_strategy_is_runnable_but_not_a_default_row():
    """``forecast_mpc`` must be asked for, never reported as a canonical row.

    It is the only strategy whose answer depends on a free parameter -- how
    wrong the forecast is -- so a fixed table row for it would be quoting one
    error level while looking like a property of the controller.
    """
    assert "forecast_mpc" in KNOWN_STRATEGIES
    assert "forecast_mpc" not in STRATEGY_ORDER


def test_the_default_belief_is_a_forecast_that_can_be_wrong():
    """The default must not quietly be perfect foresight wearing another name."""
    truth = np.linspace(0.0, 50.0, 8760)
    forecast = default_forecast_factory(truth)
    meta = forecast.metadata()
    assert meta["kind"] == "noisy"
    assert meta["parameters"]["sigma_24h"] > 0.0
    assert not np.array_equal(forecast.horizon(0, 8760), truth)
