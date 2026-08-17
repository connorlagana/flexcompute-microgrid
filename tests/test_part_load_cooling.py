"""Part-load cooling (ASSUMPTIONS Q3).

Upstream makes cooling power strictly proportional to IT power, so throttling
the GPUs to 50% halves the cooling too. Real facilities have chillers, pumps and
fans with fixed and near-fixed components, so PUE *rises* at part load. The
proportional assumption therefore over-credits throttling — it biases results in
favour of this project's hypothesis, which is exactly why it needed fixing.

``cooling_fixed_fraction`` splits nameplate cooling into a fixed and a
proportional part. Zero reproduces upstream exactly; the tests below pin both
that reproduction and the direction of the correction.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.control import FixedLoadController, SimpleThrottleController
from flexcompute.dispatch import DEFAULT_COOLING_FIXED_FRACTION, simulate, simulate_cyclic
from flexcompute.experiments import Sizing, run_strategy
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot

from conftest import BASELINE_SCENARIO, requires_weather

SNAPSHOT_PATH = SNAPSHOT_DIR / f"{BASELINE_SCENARIO.label()}.json"

requires_snapshot = pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(), reason="run scripts/run_baseline.py first"
)


@pytest.fixture(scope="module")
def base_sizing():
    if not SNAPSHOT_PATH.exists():
        pytest.skip("no baseline snapshot")
    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    return Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"])


def test_default_is_the_upstream_proportional_model():
    """The default must stay 0.0 or the Milestone 1 gate silently breaks."""
    assert DEFAULT_COOLING_FIXED_FRACTION == 0.0


@requires_weather
@requires_snapshot
def test_zero_fraction_is_bit_identical_to_the_default_path(site, base_sizing):
    common = dict(solar_mw=base_sizing.solar_mw, battery_mw=base_sizing.battery_mw,
                  battery_duration_h=base_sizing.duration_h)
    a = simulate(site, SimpleThrottleController(), **common)
    b = simulate(site, SimpleThrottleController(), cooling_fixed_fraction=0.0, **common)
    for column in ("cooling_load_mw", "bus_load_mw", "delivered_it_mw",
                   "battery_soc_mwh", "unmet_load_mw", "compute_units"):
        assert np.array_equal(a.series(column), b.series(column)), column


@requires_weather
@requires_snapshot
def test_cooling_and_bus_load_are_unchanged_at_full_power(site, base_sizing):
    """At nameplate power the two cooling models agree by construction.

    A facility running flat out pays the same cooling either way; the split only
    shows up once something throttles. If this drifts, the fixed/proportional
    decomposition is not anchored at nameplate.
    """
    common = dict(solar_mw=base_sizing.solar_mw, battery_mw=base_sizing.battery_mw,
                  battery_duration_h=base_sizing.duration_h)
    a = simulate(site, FixedLoadController(), cooling_fixed_fraction=0.0, **common)
    b = simulate(site, FixedLoadController(), cooling_fixed_fraction=0.4, **common)
    np.testing.assert_allclose(
        a.series("cooling_load_mw"), b.series("cooling_load_mw"), rtol=1e-12
    )
    np.testing.assert_allclose(
        a.series("bus_load_mw"), b.series("bus_load_mw"), rtol=1e-12
    )


@requires_weather
@requires_snapshot
def test_fixed_cooling_makes_a_brownout_hurt_the_gpus_more(site, base_sizing):
    """Non-sheddable overhead is served first, so the GPUs absorb the shortfall.

    Even a fixed load -- which never throttles -- produces less compute once
    part of its cooling cannot be shed, because during a brownout the fixed
    cooling still takes its share of the scarce power and what is left for the
    GPUs falls faster than proportionally.
    """
    common = dict(solar_mw=base_sizing.solar_mw, battery_mw=base_sizing.battery_mw,
                  battery_duration_h=base_sizing.duration_h)
    proportional = simulate(site, FixedLoadController(), cooling_fixed_fraction=0.0, **common)
    partly_fixed = simulate(site, FixedLoadController(), cooling_fixed_fraction=0.4, **common)

    assert proportional.metrics["involuntary_shortfall_mwh"] > 0     # there are brownouts
    assert partly_fixed.metrics["compute_units"] < proportional.metrics["compute_units"]
    assert (
        partly_fixed.metrics["involuntary_shortfall_mwh"]
        > proportional.metrics["involuntary_shortfall_mwh"]
    )


@requires_weather
@requires_snapshot
@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5])
def test_throttling_saves_less_cooling_when_some_of_it_is_fixed(site, base_sizing, fraction):
    common = dict(solar_mw=base_sizing.solar_mw, battery_mw=base_sizing.battery_mw,
                  battery_duration_h=base_sizing.duration_h)
    proportional = simulate(site, SimpleThrottleController(), cooling_fixed_fraction=0.0, **common)
    partly_fixed = simulate(site, SimpleThrottleController(),
                            cooling_fixed_fraction=fraction, **common)

    throttled = proportional.series("controller_target_it_mw") < (
        proportional.series("unconstrained_demand_mw") - 1e-9
    )
    assert throttled.any()
    # Wherever the fleet backs off, the fixed component keeps cooling higher.
    assert (
        partly_fixed.series("cooling_load_mw")[throttled].sum()
        > proportional.series("cooling_load_mw")[throttled].sum()
    )


@requires_weather
@requires_snapshot
def test_cooling_never_exceeds_nameplate_or_goes_negative(site, base_sizing):
    run = simulate_cyclic(
        site, SimpleThrottleController(), solar_mw=base_sizing.solar_mw,
        battery_mw=base_sizing.battery_mw, battery_duration_h=base_sizing.duration_h,
        cooling_fixed_fraction=0.35,
    )
    cooling = run.series("cooling_load_mw")
    nameplate_cooling = run.series("nameplate_it_mw") * (run.series("pue") - 1.0)
    assert cooling.min() >= -1e-9
    assert np.all(cooling <= nameplate_cooling + 1e-9)


@requires_weather
@requires_snapshot
def test_planner_and_simulator_still_agree_with_fixed_cooling(site, base_sizing):
    """Bus load becomes affine rather than proportional; the LP must follow.

    This is the tripwire for the change: if the planner kept the proportional
    form while the dispatcher moved to affine, the MPC would be optimising a
    plant that does not exist and nothing else would notice.
    """
    for fraction in (0.2, 0.4):
        run = run_strategy(
            site, "perfect_foresight_annual", base_sizing.scaled(0.4),
            cooling_fixed_fraction=fraction,
        )
        assert run.metrics["planner_model_gap"] == pytest.approx(0.0, abs=1e-6)


@requires_weather
@requires_snapshot
def test_fixed_cooling_reduces_the_measured_advantage(site, base_sizing):
    """The correction must run *against* the hypothesis, not for it.

    If a more realistic cooling model made flexible operation look better, the
    model would be wrong: non-sheddable overhead can only reduce what throttling
    saves.
    """
    sizing = base_sizing.scaled(0.25)

    def advantage(fraction: float) -> float:
        fixed = run_strategy(site, "fixed_load", sizing,
                             cooling_fixed_fraction=fraction).metrics["compute_units"]
        ceiling = run_strategy(site, "perfect_foresight_annual", sizing,
                               cooling_fixed_fraction=fraction).metrics["compute_units"]
        return ceiling / fixed - 1.0

    assert advantage(0.4) < advantage(0.0)
    assert advantage(0.4) > 0.0     # but it does not vanish


def test_invalid_fractions_are_rejected(site, base_sizing):
    with pytest.raises(ValueError, match="cooling_fixed_fraction"):
        simulate(site, FixedLoadController(), solar_mw=100.0, battery_mw=50.0,
                 cooling_fixed_fraction=1.0)
    with pytest.raises(ValueError, match="cooling_fixed_fraction"):
        simulate(site, FixedLoadController(), solar_mw=100.0, battery_mw=50.0,
                 cooling_fixed_fraction=-0.1)
