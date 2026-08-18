"""The Handmer-style governor, and the ephemeris it runs on.

Two things are being defended here.

**Faithfulness.** The governor reproduces a rule described in prose, so the
tests pin the behaviours the prose actually specifies -- ration to dawn, react
to present generation, never consult a forecast -- rather than pinning output
numbers, which would only assert that our implementation equals itself.

**Causality.** This is the first controller that reads a clock, and a clock is
one small step from a forecast. The tests below establish that the step was not
taken: the governor's decisions are invariant to every future value of solar,
and the object it is handed has no field capable of carrying one.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from flexcompute.control import (
    CaseyGovernor,
    ComputeController,
    FixedLoadController,
    Observation,
    PlantConstants,
    SimpleThrottleController,
)
from flexcompute.dispatch import simulate
from flexcompute.experiments import Sizing, run_strategy
from flexcompute.solar_clock import (
    build_solar_clock,
    daylight_mask,
    hours_until_next_sunrise,
    solar_elevation_deg,
)
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot

from conftest import BASELINE_SCENARIO, requires_weather

SNAPSHOT_PATH = SNAPSHOT_DIR / f"{BASELINE_SCENARIO.label()}.json"

requires_snapshot = pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(), reason="run scripts/run_baseline.py first"
)

HOURS = 8760


# ---------------------------------------------------------------------------
# The ephemeris
# ---------------------------------------------------------------------------

def _toy_elevation(days: int = 5) -> np.ndarray:
    """A clean day/night cycle: sun up 06:00-17:59 every day."""
    day = np.array([-10.0] * 6 + [30.0] * 12 + [-10.0] * 6)
    return np.tile(day, days)


def test_sunrise_is_always_strictly_in_the_future():
    hours = hours_until_next_sunrise(_toy_elevation())
    assert np.all(hours >= 1.0)


def test_hours_to_sunrise_counts_down_through_the_night():
    """At 18:00 the sun set; the next sunrise is 12 hours away, then 11, ..."""
    hours = hours_until_next_sunrise(_toy_elevation())
    night = hours[18:30]                      # 18:00 through 05:00 next day
    np.testing.assert_array_equal(night, np.arange(12, 0, -1))


def test_daytime_points_at_tomorrows_sunrise_not_at_zero():
    """The rule that makes rationing work during an overcast day.

    Returning zero while the sun is geometrically up would let the governor
    spend freely through a dark day and meet dusk empty.
    """
    hours = hours_until_next_sunrise(_toy_elevation())
    assert hours[6] == 24.0        # just after sunrise -> tomorrow's
    assert hours[12] == 18.0       # midday -> 18 h to tomorrow's sunrise
    assert np.all(hours[6:18] > 0)


def test_clock_wraps_at_the_year_boundary():
    """Legitimate for an ephemeris: January's sunrise is known in December."""
    hours = hours_until_next_sunrise(_toy_elevation(days=3))
    assert np.isfinite(hours).all()
    assert hours[-1] == 7.0        # 23:00 on the last day -> 06:00 on the first


def test_a_series_with_no_sunrise_is_rejected():
    with pytest.raises(ValueError):
        hours_until_next_sunrise(np.full(48, -10.0))


@requires_weather
def test_dallas_ephemeris_is_physically_plausible(site):
    clock = build_solar_clock(site)
    assert clock.elevation_deg.shape == (HOURS,)
    # Half a year of daylight, give or take refraction and the hourly grid.
    assert 4300 <= clock.daylight_hours <= 4500
    assert clock.hours_to_sunrise.min() >= 1.0
    assert clock.hours_to_sunrise.max() <= 25.0
    # Counting down 24 -> 1 once per day, so the annual mean sits near 12.5.
    assert 12.0 <= clock.hours_to_sunrise.mean() <= 13.0


@requires_weather
def test_ephemeris_ignores_weather_entirely(site):
    """Scramble the irradiance; the clock must not move.

    This is the load-bearing test for the claim that "hours until sunrise" is a
    calendar rather than a forecast.
    """
    clock = build_solar_clock(site)

    scrambled = dataclasses.replace(site.tmy, data=site.tmy.data.sample(
        frac=1.0, random_state=0
    ).set_index(site.tmy.data.index))
    other = build_solar_clock(dataclasses.replace(site, tmy=scrambled))

    np.testing.assert_array_equal(clock.elevation_deg, other.elevation_deg)
    np.testing.assert_array_equal(clock.hours_to_sunrise, other.hours_to_sunrise)


def test_elevation_requires_a_localised_index():
    import pandas as pd

    naive = pd.date_range("2023-06-01", periods=24, freq="h")
    with pytest.raises(ValueError, match="timezone-aware"):
        solar_elevation_deg(32.78, -96.80, naive)


# ---------------------------------------------------------------------------
# The rationing rule, on hand-checkable numbers
# ---------------------------------------------------------------------------

#: A trivial plant: one MW at the bus per MW of IT, no losses, no cooling.
UNIT_PLANT = PlantConstants(
    m_it=1.0, m_cool=1.0, m_solar_bus=1.0, m_batt_bus=1.0, cooling_fixed_fraction=0.0
)


def _obs(**kw) -> Observation:
    base = dict(
        t=0, hour_of_day=0, soc_mwh=100.0, soc_fraction=0.5,
        battery_energy_mwh=200.0, battery_power_mw=50.0, solar_dc_mw=0.0,
        pue=1.0, nameplate_it_mw=10.0, unconstrained_demand_mw=10.0,
        min_operating_it_mw=1.0,
    )
    base.update(kw)
    return Observation(**base)


def _governor(hours_to_sunrise=10.0, **kw) -> CaseyGovernor:
    return CaseyGovernor(
        plant=UNIT_PLANT, hours_to_sunrise=np.full(HOURS, hours_to_sunrise), **kw
    )


def test_rations_stored_energy_evenly_over_the_hours_to_sunrise():
    """100 MWh, 10 hours of darkness left, PUE 1, no losses -> 10 MW."""
    assert _governor(10.0).choose_power(_obs()) == pytest.approx(10.0)


def test_a_longer_night_means_a_deeper_throttle():
    """Same stored energy spread over 20 hours buys half the power."""
    assert _governor(20.0).choose_power(_obs()) == pytest.approx(5.0)


def test_never_asks_for_more_work_than_exists():
    """Plenty of energy for the night; the cap is demand, not the battery."""
    assert _governor(2.0).choose_power(_obs()) == pytest.approx(10.0)


def test_present_generation_is_added_to_the_ration():
    """4 MW of sun plus 100 MWh over 20 hours = 4 + 5 = 9 MW."""
    got = _governor(20.0).choose_power(_obs(solar_dc_mw=4.0))
    assert got == pytest.approx(9.0)


def test_rationing_stops_once_the_sun_carries_the_plant():
    """Sun covering full load releases the battery limit entirely.

    Without this the governor would keep throttling on a bright morning
    purely because tomorrow's sunrise is 20 hours away.
    """
    lean = _obs(soc_mwh=1.0, solar_dc_mw=10.0)
    assert _governor(20.0).choose_power(lean) == pytest.approx(10.0)


def test_battery_power_rating_caps_the_ration():
    """A short night cannot spend the battery faster than it can discharge."""
    obs = _obs(soc_mwh=200.0, battery_power_mw=3.0)
    assert _governor(2.0).choose_power(obs) == pytest.approx(3.0)


def test_parks_rather_than_operating_below_the_measured_floor():
    """Below the curve's domain there is no performance data to stand on."""
    starved = _obs(soc_mwh=1.0, min_operating_it_mw=3.0)
    assert _governor(20.0).choose_power(starved) == 0.0


def test_a_reserve_is_held_back_from_the_nightly_ration():
    plain = _governor(10.0).choose_power(_obs())
    cautious = _governor(10.0, reserve_fraction=0.25).choose_power(_obs())
    # 100 MWh stored, 25% of 200 MWh held back -> 50 MWh spendable.
    assert cautious == pytest.approx(5.0)
    assert cautious < plain


def test_power_is_monotone_in_stored_energy():
    governor = _governor(20.0)
    powers = [governor.choose_power(_obs(soc_mwh=s)) for s in np.linspace(0, 200, 40)]
    assert np.all(np.diff(powers) >= -1e-12)


def test_cooling_floor_is_subtracted_before_the_gpus_are_fed():
    """With non-sheddable cooling, some of the ration never reaches the GPUs."""
    plant = dataclasses.replace(UNIT_PLANT, cooling_fixed_fraction=0.5)
    governor = CaseyGovernor(plant=plant, hours_to_sunrise=np.full(HOURS, 10.0))
    # PUE 1.2 on 10 MW nameplate -> 2 MW cooling, half of it fixed = 1 MW floor.
    obs = _obs(pue=1.2)
    # 100 MWh / 10 h = 10 MW at the bus, minus the 1 MW floor, over
    # alpha = 1 + 0.5 * 2 / 10 = 1.1  ->  (10 - 1) / 1.1
    assert governor.choose_power(obs) == pytest.approx(9.0 / 1.1)


def test_rejects_a_non_positive_rationing_horizon():
    with pytest.raises(ValueError, match="strictly positive"):
        CaseyGovernor(plant=UNIT_PLANT, hours_to_sunrise=np.zeros(HOURS))


def test_reset_rejects_a_clock_shorter_than_the_run():
    governor = CaseyGovernor(plant=UNIT_PLANT, hours_to_sunrise=np.full(10, 5.0))
    with pytest.raises(ValueError, match="solar clock covers"):
        governor.reset(horizon=HOURS)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

def test_the_governor_cannot_be_handed_a_forecast():
    """Structural, not behavioural: there is no field for one.

    ``PlantConstants`` holds scalars only, so nothing the governor receives at
    construction is capable of carrying a time series -- of solar or anything
    else.
    """
    fields = {f.name for f in dataclasses.fields(CaseyGovernor)}
    assert "forecast" not in fields
    for field in dataclasses.fields(PlantConstants):
        assert field.type is not np.ndarray
        value = getattr(UNIT_PLANT, field.name)
        assert not isinstance(value, (np.ndarray, list, tuple, dict)), field.name


@requires_weather
@requires_snapshot
def test_decisions_are_invariant_to_the_future(site):
    """Replace every future solar value with noise; the actions must not move.

    The governor is stepped by hand here rather than through the simulator, so
    that the only thing varying between the two runs is information about hours
    the controller has not reached yet.
    """
    clock = build_solar_clock(site)
    governor = CaseyGovernor(
        plant=UNIT_PLANT, hours_to_sunrise=clock.hours_to_sunrise
    )
    truth = site.solar_p_dc * 100.0
    rng = np.random.default_rng(0)
    lies = rng.uniform(0.0, 100.0, size=HOURS)

    honest, deceived = [], []
    for t in range(0, HOURS, 37):        # sample the year; 8760 solves is slow
        soc = 40.0 + 60.0 * ((t * 7919) % 101) / 100.0
        common = dict(t=t, hour_of_day=t % 24, soc_mwh=soc, soc_fraction=soc / 200.0)
        honest.append(governor.choose_power(_obs(solar_dc_mw=truth[t], **common)))
        # Same present, entirely different future.
        future = np.concatenate([truth[: t + 1], lies[t + 1 :]])
        deceived.append(governor.choose_power(_obs(solar_dc_mw=future[t], **common)))

    np.testing.assert_array_equal(honest, deceived)


def test_governor_is_deterministic():
    governor = _governor(12.0)
    obs = _obs(soc_mwh=77.0)
    assert governor.choose_power(obs) == governor.choose_power(obs)


def test_governor_satisfies_the_controller_protocol():
    assert isinstance(_governor(), ComputeController)


def test_metadata_names_its_source_and_its_approximations():
    meta = _governor().metadata()
    assert meta["name"] == "casey_governor"
    assert "Handmer" in meta["source"]["author"]
    assert "caseyhandmer" in meta["source"]["url"]
    assert "B14" in meta["caveat"]
    assert "forecast" in meta["caveat"].lower()


# ---------------------------------------------------------------------------
# End to end, against the other controllers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scarce(site):
    """A sizing where stored energy actually binds."""
    s = load_snapshot(SNAPSHOT_PATH)["optimized"]["sizing"]
    return Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"]).scaled(0.25)


@requires_weather
@requires_snapshot
def test_governor_beats_doing_nothing_when_energy_is_scarce(site, scarce):
    """Both halves of the claim: more compute *and* less brownout."""
    fixed = run_strategy(site, "fixed_load", scarce).metrics
    casey = run_strategy(site, "casey_governor", scarce).metrics

    assert casey["compute_units"] > fixed["compute_units"]
    assert casey["involuntary_shortfall_mwh"] < 0.25 * fixed["involuntary_shortfall_mwh"]


@requires_weather
@requires_snapshot
def test_governor_beats_the_soc_only_heuristic(site, scarce):
    """Knowing the time of day is worth something over knowing only the battery."""
    throttle = run_strategy(site, "simple_throttle", scarce).metrics
    casey = run_strategy(site, "casey_governor", scarce).metrics
    assert casey["compute_units"] > throttle["compute_units"]


@requires_weather
@requires_snapshot
def test_governor_stays_under_the_perfect_foresight_ceiling(site, scarce):
    """A forecast-free rule cannot beat the optimum that knows the whole year.

    If it ever does, it is buying the difference with brownouts, and the
    shortfall column is what exposes that.
    """
    ceiling = run_strategy(site, "perfect_foresight_annual", scarce).metrics
    casey = run_strategy(site, "casey_governor", scarce).metrics
    assert casey["compute_units"] <= ceiling["compute_units"] + 1e-6


@requires_weather
@requires_snapshot
def test_governor_respects_every_physical_limit(site, scarce):
    from flexcompute.dispatch import simulate_cyclic

    result = run_strategy(site, "casey_governor", scarce)
    soc = result.series("battery_soc_mwh")
    assert soc.min() >= -1e-9
    assert soc.max() <= scarce.battery_mwh + 1e-9
    assert result.series("battery_charge_mw").max() <= scarce.battery_mw + 1e-9
    assert result.series("battery_discharge_mw").max() <= scarce.battery_mw + 1e-9
    # Target never exceeds demand, delivered never exceeds target.
    target = result.series("controller_target_it_mw")
    demand = result.series("unconstrained_demand_mw")
    delivered = result.series("delivered_it_mw")
    assert np.all(target <= demand + 1e-9)
    assert np.all(delivered <= target + 1e-9)
