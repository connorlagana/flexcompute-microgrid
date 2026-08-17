"""Physical-validity tests for the dispatch model.

These check laws, not values: energy is conserved, state stays inside its box,
power stays under its rating. They must keep passing when a controller starts
choosing the load, which is the whole reason they exist now rather than later.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.baseline import evaluate_fixed_sizing, multipliers_for
from flexcompute.metrics import audit_energy_balance

from conftest import requires_weather

MULTS = dict(solar_to_bus_mult=1.25, solar_to_battery_mult=1.30, battery_to_bus_mult=1.20)


def _dispatch(**kwargs):
    from pvstoragesim import simulate_battery_operation

    return simulate_battery_operation(**{**MULTS, **kwargs})


# ---------------------------------------------------------------------------
# Synthetic dispatch: laws hold on adversarial inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_random_dispatch_conserves_energy_and_respects_limits(seed):
    """Random solar and load, checked hour by hour against all five laws."""
    rng = np.random.default_rng(seed)
    n = 720
    solar_dc = np.clip(rng.normal(30, 40, n), 0, None)
    bus_load = np.clip(rng.normal(12, 4, n), 0.1, None)
    power_mw = float(rng.uniform(2, 40))
    energy_mwh = power_mw * float(rng.uniform(0.5, 8.0))

    out = _dispatch(
        solar_dc_mw=solar_dc,
        battery_power_mw=power_mw,
        battery_energy_mwh=energy_mwh,
        hourly_bus_load_mw=bus_load,
        initial_soc=float(rng.uniform(0, 100)),
    )
    soc = out["battery_soc"]
    charge, discharge = out["battery_charge_mw"], out["battery_discharge_mw"]
    curtailed, unmet = out["curtailed_solar_mw"], out["unmet_load_mw"]
    solar_at_bus = out["solar_at_load_mw"]

    # State bounds
    assert soc.min() >= -1e-9
    assert soc.max() <= energy_mwh + 1e-9
    # Power limits
    assert charge.max() <= power_mw + 1e-9
    assert discharge.max() <= power_mw + 1e-9
    # No negative flows
    for name, arr in [("charge", charge), ("discharge", discharge),
                      ("curtailed", curtailed), ("unmet", unmet)]:
        assert arr.min() >= -1e-9, name
    # A battery never charges and discharges in the same hour
    assert not np.any((charge > 1e-12) & (discharge > 1e-12))

    surplus = solar_at_bus > bus_load
    # PV accounting, surplus hours
    booked = bus_load * MULTS["solar_to_bus_mult"] + charge * MULTS["solar_to_battery_mult"] + curtailed
    np.testing.assert_allclose(solar_dc[surplus], booked[surplus], rtol=1e-9, atol=1e-9)
    # Bus balance, deficit hours
    supplied = solar_at_bus + discharge / MULTS["battery_to_bus_mult"] + unmet
    np.testing.assert_allclose(bus_load[~surplus], supplied[~surplus], rtol=1e-9, atol=1e-9)
    # Battery state transition
    np.testing.assert_allclose(np.diff(soc), (charge - discharge)[:-1], rtol=1e-9, atol=1e-9)


def test_battery_cannot_create_energy_over_a_full_cycle():
    """Discharged energy at the bus is strictly less than the DC energy spent
    charging: round-trip and converter losses must both bite."""
    solar_dc = np.concatenate([np.full(12, 100.0), np.zeros(12)])
    bus_load = np.concatenate([np.zeros(12), np.full(12, 5.0)])
    out = _dispatch(
        solar_dc_mw=solar_dc,
        battery_power_mw=50.0,
        battery_energy_mwh=200.0,
        hourly_bus_load_mw=bus_load,
        initial_soc=0.0,
    )
    dc_spent_charging = out["battery_charge_mw"].sum() * MULTS["solar_to_battery_mult"]
    delivered_at_bus = (out["battery_discharge_mw"] / MULTS["battery_to_bus_mult"]).sum()
    assert 0 < delivered_at_bus < dc_spent_charging


def test_zero_battery_still_balances():
    """Degenerate sizing must not divide by zero or invent storage."""
    out = _dispatch(
        solar_dc_mw=np.array([0.0, 50.0, 0.0]),
        battery_power_mw=0.0,
        battery_energy_mwh=0.0,
        hourly_bus_load_mw=np.array([10.0, 10.0, 10.0]),
        initial_soc=50.0,
    )
    assert np.all(out["battery_soc"] == 0.0)
    assert np.all(out["battery_charge_mw"] == 0.0)
    assert np.all(out["battery_discharge_mw"] == 0.0)
    assert out["unmet_load_mw"][0] == pytest.approx(10.0)


def test_inverter_cap_during_deficit_strands_pv_uncounted():
    """Known upstream accounting gap, pinned deliberately.

    When the solar->load path is inverter-capped *and* the bus is still in
    deficit, PV above the cap is neither delivered, stored, nor booked as
    curtailment. It is counted in ``solar_generation_mwh`` but appears nowhere
    in the sinks, so the annual books do not close.

    Upstream calls this "a thin edge case in an overbuilt islanded system" and
    it does not occur at the baseline sizing (measured: 0 MWh). It *can* occur
    at small solar sizings, so it is pinned here: if a future controller drives
    the system into this regime, this test documents what the number means.
    """
    out = _dispatch(
        solar_dc_mw=np.array([20.0]),
        battery_power_mw=5.0,
        battery_energy_mwh=20.0,
        hourly_bus_load_mw=np.array([12.0]),
        inverter_cap_mw_dc=10.0,
        initial_soc=100.0,
    )
    assert out["curtailed_solar_mw"][0] == 0.0          # nothing booked
    assert out["solar_at_load_mw"][0] == pytest.approx(10.0 / 1.25)
    stranded = 20.0 - 10.0                               # DC above the cap
    assert stranded == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Real site: the audit must come back clean
# ---------------------------------------------------------------------------

@requires_weather
@pytest.mark.parametrize(
    "solar_mult,battery_mult", [(6.0, 2.0), (10.0, 3.0), (15.0, 7.0), (28.0, 14.0)]
)
def test_baseline_site_dispatch_is_physical(site, solar_mult, battery_mult):
    design = site.facility_load.facility_load_design_mw
    run = evaluate_fixed_sizing(site, solar_mult * design, battery_mult * design)
    assert run.audit.violations() == []


@requires_weather
def test_baseline_site_has_no_unbooked_pv(site):
    """At baseline sizings the inverter-cap gap above does not trigger."""
    design = site.facility_load.facility_load_design_mw
    run = evaluate_fixed_sizing(site, 15.0 * design, 7.0 * design)
    assert run.audit.unbooked_pv_mwh == pytest.approx(0.0, abs=1e-9)


@requires_weather
def test_independent_battery_duration_is_honoured(site):
    """Passing a non-4-hour duration really does change the energy capacity.

    Upstream already supports this on ``evaluate_system``; only the optimiser
    hard-codes 4 h. Confirming it here means Experiment B's decoupling of MW
    from MWh does not need a dispatch rewrite.
    """
    design = site.facility_load.facility_load_design_mw
    battery_mw = 7.0 * design
    two_hour = evaluate_fixed_sizing(site, 15.0 * design, battery_mw, battery_duration_h=2.0)
    eight_hour = evaluate_fixed_sizing(site, 15.0 * design, battery_mw, battery_duration_h=8.0)

    assert two_hour.metrics["battery_mwh"] == pytest.approx(battery_mw * 2.0)
    assert eight_hour.metrics["battery_mwh"] == pytest.approx(battery_mw * 8.0)
    assert eight_hour.metrics["uptime_pct"] > two_hour.metrics["uptime_pct"]
    assert two_hour.audit.violations() == []
    assert eight_hour.audit.violations() == []


@requires_weather
def test_energy_sinks_sum_to_generation(site):
    """Annual books close: generation = delivered + charged + curtailed + losses."""
    design = site.facility_load.facility_load_design_mw
    run = evaluate_fixed_sizing(site, 15.0 * design, 7.0 * design)
    mult = multipliers_for(site)
    hourly = run.sim.hourly_data

    generation = run.metrics["solar_generation_mwh"]
    bus_load = (
        hourly["it_load_mw"].to_numpy() * mult["bus_to_it"]
        + hourly["cooling_load_mw"].to_numpy() * mult["bus_to_cooling"]
    )
    solar_at_bus = hourly["solar_at_load_mw"].to_numpy()
    dc_to_load = np.minimum(solar_at_bus, bus_load).sum() * mult["solar_to_bus"]
    dc_to_battery = run.metrics["battery_charged_mwh"] * mult["solar_to_battery"]
    curtailed = run.metrics["solar_curtailed_mwh"]

    assert dc_to_load + dc_to_battery + curtailed == pytest.approx(generation, rel=1e-9)
