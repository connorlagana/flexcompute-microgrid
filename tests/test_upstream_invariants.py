"""Pin the upstream behaviours this project builds on top of.

These are not tests of upstream's correctness -- they are tripwires. Each one
encodes an assumption our code or our experimental design relies on. If the
vendored reference model is ever updated and one of these fails, the failure
message says which downstream design decision needs revisiting.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from flexcompute.baseline import UPSTREAM_BATTERY_DURATION_H


# ---------------------------------------------------------------------------
# Load entry point
# ---------------------------------------------------------------------------

def test_it_load_is_a_plain_hourly_array():
    """The IT load enters dispatch as one 8760 array on FacilityLoad.

    This is the seam the ComputeController will replace. If upstream ever
    computes IT load inside the dispatch loop instead, the controller design
    has to change.
    """
    from it_facil import FacilityLoad

    fields = FacilityLoad.__dataclass_fields__
    assert "hourly_it_load_mw" in fields
    assert "hourly_cooling_load_mw" in fields
    assert "hourly_pue" in fields


def test_cooling_load_is_a_fixed_multiple_of_it_load():
    """cooling = it * (pue - 1), with PUE set by weather alone.

    Our controller work depends on knowing exactly how cooling responds when
    GPU power is throttled. Upstream's answer is 'linearly, with no fixed
    overhead' -- an assumption we inherit and must document, not discover
    later. See ASSUMPTIONS.md ("PUE is load-independent").
    """
    from it_facil import calculate_facility_load

    fl = calculate_facility_load(total_gpus=800, use_hourly_load_csv=False, pue=1.2)
    it = np.asarray(fl.hourly_it_load_mw)
    cooling = np.asarray(fl.hourly_cooling_load_mw)
    np.testing.assert_allclose(cooling, it * (1.2 - 1.0), rtol=1e-12)


def test_pue_lookup_takes_only_weather():
    """PUE is a function of (temperature, humidity) -- not of IT load."""
    from pue_tool import calculate_annual_pue

    params = inspect.signature(calculate_annual_pue).parameters
    assert set(params) == {"weather_df", "lookup_df"}


# ---------------------------------------------------------------------------
# Dispatch assumptions we intend to change later
# ---------------------------------------------------------------------------

def test_battery_energy_is_derived_from_power_via_fixed_duration():
    """MW and MWh are not independent upstream: MWh = MW x 4 h.

    Experiment B wants to size power and energy separately. This pins the
    coupling so the change is explicit when we make it.
    """
    from pvstoragesim import evaluate_system

    default = inspect.signature(evaluate_system).parameters["battery_duration_hours"].default
    assert default == UPSTREAM_BATTERY_DURATION_H == 4.0

    from microgrid_optimizer import SystemCosts

    assert SystemCosts.__dataclass_fields__["battery_hours"].default == 4.0


def test_battery_cost_has_no_energy_term():
    """BESS capex is quoted per kW only, for an implied 4-hour system.

    Consequence: an optimiser allowed to vary duration under this cost model
    would buy unlimited MWh for free. Decomposing $/kW and $/kWh is a
    prerequisite for Experiment B, not an optional refinement.
    """
    from config import CostConfig
    from microgrid_optimizer import SystemCosts

    assert "bess_cost_y0" in CostConfig.__dataclass_fields__
    # No storage-energy price anywhere: every battery cost term is per kW.
    battery_fields = [
        f for f in CostConfig.__dataclass_fields__
        if "bess" in f.lower() or "battery" in f.lower() or "storage" in f.lower()
    ]
    assert battery_fields
    assert not any("kwh" in f.lower() or "mwh" in f.lower() for f in battery_fields)

    # And the cost function itself only sees MW.
    params = set(inspect.signature(SystemCosts.calculate_system_cost).parameters)
    assert params == {"self", "solar_mw", "battery_mw"}


def test_initial_soc_is_hardcoded_at_75_percent():
    """evaluate_system starts the year with a 75%-full battery, for free.

    The simulation is never required to return that energy, so a marginal
    design is credited with stored energy nobody generated. Measured effect
    for the Dallas baseline is in ASSUMPTIONS.md.
    """
    from pvstoragesim import evaluate_system

    source = inspect.getsource(evaluate_system)
    assert "initial_soc=75.0" in source

    from pvstoragesim import simulate_battery_operation

    assert inspect.signature(simulate_battery_operation).parameters["initial_soc"].default == 50


def test_uptime_counts_hours_with_unmet_below_one_kilowatt():
    """Uptime is an absolute-threshold hour count, not an energy fraction.

    A throttled-but-online data centre would score full uptime here, which is
    precisely why useful compute -- not uptime -- has to be the project's
    headline metric.
    """
    from pvstoragesim import evaluate_system

    assert "unmet_load_mw'] < 0.001" in inspect.getsource(evaluate_system)


# ---------------------------------------------------------------------------
# Efficiency bookkeeping
# ---------------------------------------------------------------------------

def test_round_trip_efficiency_is_split_across_both_directions(cfg):
    """sqrt(RTE) on charge and sqrt(RTE) on discharge -- no free energy."""
    from power_systems_estimator import PowerFlowAnalyzer

    pfa = PowerFlowAnalyzer(cfg, topology="mv_coupled")
    charge_eff = 1.0 / pfa.get_bus_architecture_multipliers("ac_coupled")["solar_to_battery"]
    discharge_eff = 1.0 / pfa.get_bus_architecture_multipliers("ac_coupled")["battery_to_bus"]
    # Both paths carry exactly one sqrt(RTE) factor, so the product of the two
    # is bounded above by RTE itself (the rest is converter losses).
    assert charge_eff * discharge_eff < cfg.efficiency.battery_rte


@pytest.mark.parametrize("architecture", ["ac_coupled", "dc_coupled"])
@pytest.mark.parametrize("topology", ["mv_coupled", "lv_direct"])
def test_every_conversion_multiplier_is_lossy(cfg, architecture, topology):
    from power_systems_estimator import PowerFlowAnalyzer

    pfa = PowerFlowAnalyzer(cfg, topology=topology)
    for name, value in pfa.get_bus_architecture_multipliers(architecture).items():
        assert value > 1.0, f"{name} is not lossy"


# ---------------------------------------------------------------------------
# Workload shape
# ---------------------------------------------------------------------------

def test_reference_load_shape_is_essentially_flat():
    """The reference IT profile is a 24/7 training load, +/-7% around its mean.

    That is what makes the fixed-load baseline expensive: demand does not move,
    so supply must. Worth pinning, because the size of the prize in this
    project scales with how inflexible the baseline is.
    """
    from flexcompute.upstream_bridge import UPSTREAM_TABLES
    from it_facil import load_hourly_load_data

    shape = load_hourly_load_data(str(UPSTREAM_TABLES / "hourly_load_data.csv"))
    assert shape.shape == (8760,)
    assert shape.mean() == pytest.approx(1.0, abs=1e-9)
    assert 0.90 < shape.min() < 0.95
    assert 1.05 < shape.max() < 1.10
