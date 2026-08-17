"""Fixed-load baseline: reproduce the reference model, exactly and repeatably.

This is Strategy #1 of the eventual comparison -- GPU demand is exogenous, and
the supply side is sized to meet it. Nothing here introduces a controller. Its
whole purpose is to pin down the numbers that later work must not disturb.

Two entry points:

``evaluate_fixed_sizing``
    One year of dispatch at a *given* solar/battery sizing. Cheap (~30 ms),
    fully deterministic, and the workhorse for tests.

``optimize_sizing``
    Upstream's two-stage sizing search (Latin-hypercube screening -> differential
    evolution) under a fixed uptime constraint, plus the LCOE that upstream
    would report for the winner. Seeded, so it is reproducible; upstream's own
    ``compare_datacenter_power_systems`` does *not* seed it, which is why this
    module drives ``MicrogridOptimizer`` directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .metrics import EnergyAudit, audit_energy_balance, summarize_simulation
from .scenario import Site
from .upstream_bridge import ensure_upstream_importable, upstream_workdir

logger = logging.getLogger(__name__)

#: Upstream ties battery energy to battery power through a fixed duration.
#: Kept as an explicit named constant because breaking this coupling is a
#: planned change (see ASSUMPTIONS.md, "battery duration").
UPSTREAM_BATTERY_DURATION_H = 4.0


@dataclass
class BaselineRun:
    """Result of one fixed-load simulation, with its physical audit attached."""

    metrics: dict
    audit: EnergyAudit
    sim: object = field(repr=False)      # upstream SimulationResult

    def assert_physical(self, tol_mw: float = 1e-6) -> None:
        problems = self.audit.violations(tol_mw)
        if problems:
            raise AssertionError("Energy-balance audit failed:\n  " + "\n  ".join(problems))


def multipliers_for(site: Site) -> dict:
    """Power-flow multipliers (1/efficiency per directed edge) for this site."""
    ensure_upstream_importable()
    from power_systems_estimator import PowerFlowAnalyzer  # type: ignore[import-not-found]

    analyzer = PowerFlowAnalyzer(site.config, topology=site.scenario.topology)
    return analyzer.get_bus_architecture_multipliers(site.scenario.architecture)


def inverter_cap_mw_dc(site: Site, solar_mw: float) -> Optional[float]:
    """The solar->load inverter cap upstream applies, or None if uncapped.

    Mirrors the branch in ``evaluate_system``: only the AC-coupled MV-coupled
    combination caps the load path (its DC-coupled battery recaptures PV above
    the cap). AC lv_direct pre-clips the array instead, and DC architectures
    have no inverter on the PV path.
    """
    if site.scenario.architecture != "ac_coupled":
        return None
    if site.scenario.topology != "mv_coupled":
        return None
    return solar_mw / site.config.efficiency.inverter_load_ratio


def evaluate_fixed_sizing(
    site: Site,
    solar_mw: float,
    battery_mw: float,
    *,
    battery_duration_h: float = UPSTREAM_BATTERY_DURATION_H,
    audit: bool = True,
) -> BaselineRun:
    """Simulate one year at a fixed sizing, exactly as upstream would."""
    ensure_upstream_importable()
    from pvstoragesim import evaluate_system  # type: ignore[import-not-found]

    sim = evaluate_system(
        latitude=site.scenario.latitude,
        longitude=site.scenario.longitude,
        solar_capacity_mw=solar_mw,
        battery_power_mw=battery_mw,
        facility_load=site.facility_load,
        hourly_pue=site.hourly_pue,
        architecture=site.scenario.architecture,
        topology=site.scenario.topology,
        efficiency_params=site.config,
        solar_profile=site.solar_profile,
        battery_duration_hours=battery_duration_h,
        return_hourly=True,
    )
    battery_mwh = battery_mw * battery_duration_h
    metrics = summarize_simulation(
        sim, solar_mw=solar_mw, battery_mw=battery_mw, battery_mwh=battery_mwh
    )
    energy_audit = audit_energy_balance(
        sim.hourly_data,
        multipliers_for(site),
        battery_energy_mwh=battery_mwh,
        battery_power_mw=battery_mw,
        inverter_cap_mw_dc=inverter_cap_mw_dc(site, solar_mw),
    ) if audit else None
    return BaselineRun(metrics=metrics, audit=energy_audit, sim=sim)


def _system_costs(site: Site):
    """Upstream ``SystemCosts`` for this site's architecture."""
    ensure_upstream_importable()
    from microgrid_optimizer import SystemCosts  # type: ignore[import-not-found]

    costs = site.config.costs
    is_dc = site.scenario.architecture == "dc_coupled"
    return SystemCosts(
        solar_cost_per_kw=costs.solar_cost_y0,
        battery_cost_per_kw=costs.bess_cost_y0,
        solar_bos_cost_per_kw=costs.solar_bos_cost_y0_dc if is_dc else costs.solar_bos_cost_y0_ac,
        battery_bos_cost_per_kw=costs.battery_bos_cost_y0_dc if is_dc else costs.battery_bos_cost_y0_ac,
        battery_hours=UPSTREAM_BATTERY_DURATION_H,
    )


def optimize_sizing(site: Site, *, verbose: bool = False) -> dict:
    """Run upstream's seeded sizing optimisation and price the winner.

    Returns a dict with the chosen sizing, the year-0 dispatch metrics, the
    energy audit, and the LCOE upstream would report. The whole call runs with
    CWD at the upstream root because several degradation helpers are invoked
    upstream without a config argument and fall back to a root-relative path
    for the battery fade surrogate.
    """
    ensure_upstream_importable()
    from lcoe_calc import calculate_solar_storage_lcoe  # type: ignore[import-not-found]
    from microgrid_optimizer import MicrogridOptimizer   # type: ignore[import-not-found]

    started = time.time()
    with upstream_workdir():
        optimizer = MicrogridOptimizer(
            latitude=site.scenario.latitude,
            longitude=site.scenario.longitude,
            facility_load=site.facility_load,
            required_uptime_pct=site.scenario.required_uptime_pct,
            costs=_system_costs(site),
            architecture=site.scenario.architecture,
            topology=site.scenario.topology,
            efficiency_params=site.config,
            verbose=verbose,
            seed=site.scenario.seed,
        )
        # Guarantee bit-identical PV input between this search and any
        # standalone evaluation of its result.
        optimizer.solar_profile = site.solar_profile

        result = optimizer.optimize()

        lcoe = calculate_solar_storage_lcoe(
            system_type=site.scenario.architecture,
            solar_mw=result.solar_mw,
            battery_mw=result.battery_mw,
            battery_mwh=result.battery_mwh,
            land_acres=result.land_area_acres,
            sim_year_0=result.sim_year_0,
            sim_year_13=result.sim_year_13,
            sim_year_14=result.sim_year_14,
            sim_year_25=result.sim_year_25,
            year_0_stats=result.year_0_stats,
            construction_years=site.config.design.solar_construction_years,
            required_uptime_pct=site.scenario.required_uptime_pct,
            config=site.config,
        )
    elapsed = time.time() - started

    # Upstream caches simulations on a 1 MW-rounded key, so the ``sim_year_0``
    # it hands back can be a run of a *neighbouring* sizing (up to 0.5 MW away
    # in each dimension) rather than of the sizing it reports. Its own
    # feasibility test used that cached run, which is upstream's business; but
    # we must not report dispatch metrics that belong to a different system
    # than the one we quote costs for. So re-simulate the reported sizing
    # exactly, and record how far the cached run had drifted.
    verified = evaluate_fixed_sizing(site, result.solar_mw, result.battery_mw)
    verified.assert_physical()
    metrics = verified.metrics
    energy_audit = verified.audit
    cached_uptime = float(result.sim_year_0.uptime_pct)
    cache_drift = {
        "cached_uptime_pct": cached_uptime,
        "verified_uptime_pct": metrics["uptime_pct"],
        "uptime_delta_pp": metrics["uptime_pct"] - cached_uptime,
        "cached_battery_mw_implied": float(
            result.sim_year_0.hourly_data["battery_soc_mwh"].max() / UPSTREAM_BATTERY_DURATION_H
        ),
        "note": (
            "Upstream's optimizer caches on a 1 MW-rounded key; its returned "
            "sim_year_0 may belong to a sizing up to 0.5 MW from the reported "
            "one. Metrics above are re-simulated at the reported sizing."
        ),
    }

    return {
        "sizing": {
            "solar_mw_dc": float(result.solar_mw),
            "battery_mw": float(result.battery_mw),
            "battery_mwh": float(result.battery_mwh),
            "battery_duration_h": UPSTREAM_BATTERY_DURATION_H,
            "land_area_acres": float(result.land_area_acres),
        },
        "cost": {
            "capex_optimizer_objective_musd": float(result.total_cost_million),
            "lcoe_usd_per_kwh": float(lcoe.lcoe),
            "capex_npv_usd": float(lcoe.capex_npv),
            "opex_npv_usd": float(lcoe.opex_npv),
            "energy_npv_mwh": float(lcoe.energy_npv),
        },
        "year_0": metrics,
        "degradation_uptime_pct": {
            "year_0": float(result.sim_year_0.uptime_pct),
            "year_13": float(result.sim_year_13.uptime_pct),
            "year_14": float(result.sim_year_14.uptime_pct),
            "year_25": float(result.sim_year_25.uptime_pct),
        },
        "battery_year_0_stats": {k: float(v) for k, v in result.year_0_stats.items()},
        "audit": energy_audit.as_dict(),
        "optimizer_cache_drift": cache_drift,
        "optimizer": {
            "message": result.optimization_message,
            "function_evaluations": int(result.function_evaluations),
            "seed": site.scenario.seed,
            "wall_time_s": round(elapsed, 1),
        },
    }
