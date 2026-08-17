"""Closed-loop dispatch: the controller is inside the loop.

Upstream computes the whole bus-load array up front and then walks it
(`pvstoragesim.simulate_battery_operation`). That is correct for an exogenous
load and impossible for a controlled one, because the load at time ``t`` now
depends on the battery state at time ``t``, which depends on the load at
``t-1``.

This module reproduces upstream's dispatch arithmetic **exactly**, one
timestep at a time, with the bus load produced inside the loop by a
:class:`~flexcompute.control.ComputeController`.

Bit-exactness is a requirement, not an aspiration. Under ``FixedLoadController``
every array returned here is bit-identical to upstream's, which is what makes
"the controller abstraction changed nothing" a testable claim rather than a
hope. Two details matter for that:

* cooling is computed as ``it * pue - it``, matching upstream's
  ``facility_load - it_load`` subtraction, **not** the algebraically equal
  ``it * (pue - 1)`` which differs in floating point;
* the loop body, its comparisons, and its ``min`` calls are transcribed in
  upstream's order.

Do not "clean up" the arithmetic here. It is shaped this way on purpose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .control import ComputeController, Observation
from .gpu import DEFAULT_AGGREGATION, FleetAggregation, GpuFleet, PowerPerformanceCurve, get_curve
from .scenario import Site
from .upstream_bridge import ensure_upstream_importable

HOURS_PER_YEAR = 8760

#: Upstream's hard-coded starting state of charge (percent). See ASSUMPTIONS A7.
UPSTREAM_INITIAL_SOC_PCT = 75.0

#: Upstream's uptime threshold: an hour counts as "up" below 1 kW unmet at the bus.
UPTIME_UNMET_THRESHOLD_MW = 0.001

#: Fraction of nameplate cooling power that does *not* scale with IT load.
#: 0.0 reproduces upstream exactly (cooling strictly proportional to IT power).
#: See ASSUMPTIONS Q3 for why a positive value is more realistic and why it
#: works against this project's hypothesis.
DEFAULT_COOLING_FIXED_FRACTION = 0.0

#: An hour counts as "throttled" only when the target is this far below demand,
#: relatively. An absolute epsilon would let LP solver noise (order 1e-8 MW on a
#: 9 MW demand) register as deliberate throttling and inflate the statistic for
#: every optimiser-driven strategy.
THROTTLE_DETECTION_RELATIVE_TOL = 1e-6


@dataclass
class DispatchResult:
    """One simulated year under one controller."""

    hourly: pd.DataFrame
    metrics: dict
    metadata: dict = field(default_factory=dict)

    def series(self, name: str) -> np.ndarray:
        return self.hourly[name].to_numpy(dtype=float)


def simulate(
    site: Site,
    controller: ComputeController,
    *,
    solar_mw: float,
    battery_mw: float,
    battery_duration_h: float = 4.0,
    curve: Optional[PowerPerformanceCurve] = None,
    aggregation: FleetAggregation = DEFAULT_AGGREGATION,
    cooling_fixed_fraction: float = DEFAULT_COOLING_FIXED_FRACTION,
    unconstrained_demand_mw: Optional[np.ndarray] = None,
    initial_soc_pct: float = UPSTREAM_INITIAL_SOC_PCT,
) -> DispatchResult:
    """Simulate one year with the controller choosing GPU power each hour.

    ``unconstrained_demand_mw`` defaults to the site's nameplate profile, so
    demand and nameplate coincide unless a caller deliberately separates them.
    """
    ensure_upstream_importable()
    from power_systems_estimator import PowerFlowAnalyzer  # type: ignore[import-not-found]

    curve = curve or get_curve()
    fleet = GpuFleet(
        curve=curve, total_gpus=site.scenario.total_gpus, aggregation=aggregation
    )

    analyzer = PowerFlowAnalyzer(site.config, topology=site.scenario.topology)
    mult = analyzer.get_bus_architecture_multipliers(site.scenario.architecture)
    m_it = mult["bus_to_it"]
    m_cool = mult["bus_to_cooling"]
    m_solar_bus = mult["solar_to_bus"]
    m_solar_batt = mult["solar_to_battery"]
    m_batt_bus = mult["battery_to_bus"]

    nameplate = site.it_load_mw
    demand = nameplate if unconstrained_demand_mw is None else np.asarray(
        unconstrained_demand_mw, dtype=float
    )
    if demand.shape != (HOURS_PER_YEAR,):
        raise ValueError(f"unconstrained_demand_mw must have {HOURS_PER_YEAR} values")
    if not 0.0 <= cooling_fixed_fraction < 1.0:
        raise ValueError("cooling_fixed_fraction must lie in [0, 1)")
    pue = site.hourly_pue

    # Cooling power that is present regardless of GPU load (ASSUMPTIONS Q3).
    # Referenced to *nameplate* cooling, so a throttled facility still pays it.
    nameplate_cooling_mw = nameplate * pue - nameplate
    fixed_cooling_mw = cooling_fixed_fraction * nameplate_cooling_mw

    battery_energy_mwh = battery_mw * battery_duration_h
    solar_dc_mw = site.solar_p_dc * solar_mw

    # Inverter clipping, transcribed from pvstoragesim.evaluate_system.
    inverter_cap_mw_dc = None
    if site.scenario.architecture == "ac_coupled":
        inverter_limit_mw = solar_mw / site.config.efficiency.inverter_load_ratio
        if site.scenario.topology == "lv_direct":
            solar_dc_mw = np.minimum(solar_dc_mw, inverter_limit_mw)
        else:
            inverter_cap_mw_dc = inverter_limit_mw

    if inverter_cap_mw_dc is not None:
        solar_dc_to_bus = np.minimum(solar_dc_mw, inverter_cap_mw_dc)
    else:
        solar_dc_to_bus = solar_dc_mw
    solar_at_bus_mw = solar_dc_to_bus / m_solar_bus

    n = HOURS_PER_YEAR
    soc = np.zeros(n + 1)
    soc[0] = (initial_soc_pct / 100.0) * battery_energy_mwh

    target_it = np.zeros(n)
    delivered_it = np.zeros(n)
    cooling_target = np.zeros(n)
    bus_load = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    curtailed = np.zeros(n)
    unmet = np.zeros(n)
    parked = np.zeros(n, dtype=bool)

    floor_fraction = fleet.min_power_fraction
    controller.reset(horizon=n)

    for t in range(n):
        # ---- decide -----------------------------------------------------
        obs = Observation(
            t=t,
            hour_of_day=t % 24,
            soc_mwh=float(soc[t]),
            soc_fraction=float(soc[t] / battery_energy_mwh) if battery_energy_mwh > 0 else 0.0,
            battery_energy_mwh=battery_energy_mwh,
            battery_power_mw=battery_mw,
            solar_dc_mw=float(solar_dc_mw[t]),
            pue=float(pue[t]),
            nameplate_it_mw=float(nameplate[t]),
            unconstrained_demand_mw=float(demand[t]),
            min_operating_it_mw=floor_fraction * float(demand[t]),
        )
        requested = controller.choose_power(obs)
        it_mw = fleet.clamp_request(requested, demand[t])
        target_it[t] = it_mw
        parked[t] = fleet.is_parked(it_mw, demand[t])

        # ---- facility load ----------------------------------------------
        if cooling_fixed_fraction == 0.0:
            # Transcribed from it_facil: cooling = facility - it, where
            # facility = it * pue. Not it * (pue - 1): different in floating
            # point, and this branch is what the bit-exactness gate depends on.
            cooling_mw = it_mw * pue[t] - it_mw
        else:
            # Fixed + proportional. Affine in IT power, so the planner stays an
            # LP; reduces to the branch above when the fraction is zero.
            variable = (1.0 - cooling_fixed_fraction) * nameplate_cooling_mw[t]
            share = it_mw / demand[t] if demand[t] > 0 else 0.0
            cooling_mw = fixed_cooling_mw[t] + variable * share
        cooling_target[t] = cooling_mw
        load_at_bus = it_mw * m_it + cooling_mw * m_cool
        bus_load[t] = load_at_bus

        # ---- dispatch (upstream's loop body, verbatim in structure) ------
        power_balance = solar_at_bus_mw[t] - load_at_bus
        if power_balance > 0:
            dc_used_for_load = load_at_bus * m_solar_bus
            leftover_pv_dc = solar_dc_mw[t] - dc_used_for_load
            max_charge_at_battery = min(battery_mw, battery_energy_mwh - soc[t])
            charge_at_battery = min(leftover_pv_dc / m_solar_batt, max_charge_at_battery)
            charge[t] = charge_at_battery
            soc[t + 1] = soc[t] + charge_at_battery
            dc_used_for_charging = charge_at_battery * m_solar_batt
            curtailed[t] = leftover_pv_dc - dc_used_for_charging
        else:
            deficit_at_bus = -power_balance
            max_discharge_from_battery = min(battery_mw, soc[t])
            max_discharge_at_bus = max_discharge_from_battery / m_batt_bus
            discharge_at_bus = min(deficit_at_bus, max_discharge_at_bus)
            discharge_from_battery = discharge_at_bus * m_batt_bus
            discharge[t] = discharge_from_battery
            soc[t + 1] = soc[t] - discharge_from_battery
            unmet[t] = deficit_at_bus - discharge_at_bus

        # ---- what actually reached the GPUs ------------------------------
        # Bus load is affine in IT power: alpha * u + beta, where beta is the
        # fixed cooling floor. A shortfall cannot be taken out of beta -- fixed
        # cooling is fixed -- so it falls entirely on the variable part.
        served_at_bus = load_at_bus - unmet[t]
        if cooling_fixed_fraction == 0.0:
            served_fraction = (served_at_bus / load_at_bus) if load_at_bus > 0.0 else 1.0
            delivered_it[t] = it_mw * served_fraction
        else:
            beta = fixed_cooling_mw[t] * m_cool
            alpha = m_it + (1.0 - cooling_fixed_fraction) * (
                nameplate_cooling_mw[t] / demand[t] if demand[t] > 0 else 0.0
            ) * m_cool
            delivered_it[t] = (
                min(max((served_at_bus - beta) / alpha, 0.0), it_mw) if alpha > 0 else 0.0
            )

    # -- compute accounting ------------------------------------------------
    compute_units = fleet.compute_units(delivered_it, demand)
    nameplate_compute = fleet.compute_units(demand, demand)   # 1.0 wherever demand > 0

    voluntary_throttle_mw = demand - target_it
    involuntary_shortfall_mw = target_it - delivered_it

    hourly = pd.DataFrame(
        {
            "hour": np.arange(n),
            "solar_dc_mw": solar_dc_mw,
            "solar_at_bus_mw": solar_at_bus_mw,
            "pue": pue,
            # the four power quantities
            "nameplate_it_mw": nameplate,
            "unconstrained_demand_mw": demand,
            "controller_target_it_mw": target_it,
            "delivered_it_mw": delivered_it,
            # their two differences
            "voluntary_throttle_mw": voluntary_throttle_mw,
            "involuntary_shortfall_mw": involuntary_shortfall_mw,
            # facility and dispatch
            "cooling_load_mw": cooling_target,
            "bus_load_mw": bus_load,
            "battery_soc_mwh": soc[:-1],
            "battery_charge_mw": charge,
            "battery_discharge_mw": discharge,
            "curtailed_solar_mw": curtailed,
            "unmet_load_mw": unmet,
            "compute_units": compute_units,
            "parked": parked,
        }
    )

    metrics = _summarize(
        hourly,
        solar_mw=solar_mw,
        battery_mw=battery_mw,
        battery_energy_mwh=battery_energy_mwh,
        nameplate_compute_total=float(nameplate_compute.sum()),
        soc_end_mwh=float(soc[-1]),
    )

    return DispatchResult(
        hourly=hourly,
        metrics=metrics,
        metadata={
            "controller": controller.metadata(),
            "gpu": fleet.metadata(),
            "scenario": asdict(site.scenario),
            "sizing": {
                "solar_mw_dc": float(solar_mw),
                "battery_mw": float(battery_mw),
                "battery_mwh": float(battery_energy_mwh),
                "battery_duration_h": float(battery_duration_h),
            },
            "initial_soc_pct": float(initial_soc_pct),
            "initial_soc_mode": "fixed",
            "cooling_fixed_fraction": float(cooling_fixed_fraction),
        },
    )


def _summarize(
    hourly: pd.DataFrame,
    *,
    solar_mw: float,
    battery_mw: float,
    battery_energy_mwh: float,
    nameplate_compute_total: float,
    soc_end_mwh: float,
) -> dict:
    get = lambda name: hourly[name].to_numpy(dtype=float)  # noqa: E731

    unmet = get("unmet_load_mw")
    bus_load = get("bus_load_mw")
    compute = get("compute_units")
    demand = get("unconstrained_demand_mw")
    delivered = get("delivered_it_mw")
    target = get("controller_target_it_mw")
    soc = get("battery_soc_mwh")
    parked = hourly["parked"].to_numpy(dtype=bool)

    solar_generation = float(get("solar_dc_mw").sum())
    curtailed = float(get("curtailed_solar_mw").sum())
    charged = float(get("battery_charge_mw").sum())
    discharged = float(get("battery_discharge_mw").sum())

    bus_served = float((bus_load - unmet).sum())
    bus_total = float(bus_load.sum())

    throttled = int(
        ((target < demand * (1.0 - THROTTLE_DETECTION_RELATIVE_TOL)) & ~parked).sum()
    )

    return {
        # -- sizing --
        "solar_mw_dc": float(solar_mw),
        "battery_mw": float(battery_mw),
        "battery_mwh": float(battery_energy_mwh),
        # -- the headline: useful compute --
        "compute_units": float(compute.sum()),
        "compute_units_nameplate": nameplate_compute_total,
        "compute_fraction_of_nameplate": (
            float(compute.sum() / nameplate_compute_total) if nameplate_compute_total else 0.0
        ),
        # -- energy --
        "solar_generation_mwh": solar_generation,
        "solar_curtailed_mwh": curtailed,
        "solar_curtailed_pct": float(100.0 * curtailed / solar_generation) if solar_generation else 0.0,
        "gpu_energy_delivered_mwh": float(delivered.sum()),
        "bus_load_served_mwh": bus_served,
        "bus_load_requested_mwh": bus_total,
        "battery_charged_mwh": charged,
        "battery_discharged_mwh": discharged,
        "battery_cycles_per_year": float(discharged / battery_energy_mwh) if battery_energy_mwh else 0.0,
        # -- the voluntary / involuntary split --
        "voluntary_throttle_mwh": float(get("voluntary_throttle_mw").sum()),
        "involuntary_shortfall_mwh": float(get("involuntary_shortfall_mw").sum()),
        "unserved_bus_energy_mwh": float(unmet.sum()),
        "hours_throttled": throttled,
        "hours_parked": int(parked.sum()),
        "hours_with_shortfall": int((get("involuntary_shortfall_mw") > 1e-9).sum()),
        # -- reference-model reliability metrics, for comparability --
        "uptime_pct": float(100.0 * (unmet < UPTIME_UNMET_THRESHOLD_MW).sum() / len(unmet)),
        "energy_served_pct": float(100.0 * bus_served / bus_total) if bus_total else 0.0,
        # -- battery state --
        "soc_start_mwh": float(soc[0]),
        "soc_end_mwh": soc_end_mwh,
        "soc_mean_mwh": float(soc.mean()),
        "soc_min_mwh": float(soc.min()),
        "soc_net_change_mwh": float(soc_end_mwh - soc[0]),
    }


# ---------------------------------------------------------------------------
# Cyclic boundary condition (ASSUMPTIONS Q1)
# ---------------------------------------------------------------------------

#: Convergence tolerance for the cyclic SOC fixed point, as a fraction of
#: battery energy capacity.
CYCLIC_SOC_TOL_FRACTION = 1e-9

#: Relaxed acceptance for controllers whose start->end SOC map is discontinuous.
#: A bang-bang policy can have *no* exact fixed point: an arbitrarily small
#: change in starting SOC can flip a band and move the year-end state by MWh. We
#: accept a residual this small relative to capacity -- on a 490 MWh battery that
#: is ~0.05 MWh against ~90,000 MWh served, i.e. free energy at the 1e-6 level --
#: and record the actual residual so it stays auditable.
CYCLIC_SOC_RELAXED_TOL_FRACTION = 1e-4


def simulate_cyclic(
    site: Site,
    controller: ComputeController,
    *,
    max_iterations: int = 12,
    initial_soc_pct: float = UPSTREAM_INITIAL_SOC_PCT,
    **kwargs,
) -> DispatchResult:
    """Simulate with a self-sustaining year: end-of-year SOC == start-of-year SOC.

    Upstream hands the simulation a 75%-full battery for free and never asks
    for it back (ASSUMPTIONS A7), which flatters exactly the January hours that
    set the sizing constraint. The physically meaningful statement is instead
    *the year must be self-sustaining*: whatever the battery holds on 1 January
    it must hold again on 31 December.

    Two-phase solve. Plain fixed-point iteration first -- simulate, take the
    year-end SOC, use it as the next start, repeat -- which converges in two
    iterations for a smooth controller because the battery saturates mid-year
    and forgets its initial condition.

    That iteration can *cycle forever* for a bang-bang controller. Its
    start->end map is a step function, so a hair's change in starting SOC flips
    an SOC band and moves the year-end state by whole MWh; observed as a stable
    period-5 orbit for ``SimpleThrottleController`` at some sizings. Phase two
    therefore brackets the residual ``end(s) - s`` and bisects it, which is
    robust to those jumps.

    A discontinuous map may have no exact fixed point at all, so the bisection
    accepts a residual that is negligible against annual throughput rather than
    demanding zero. The residual is recorded either way; it never silently
    becomes free energy.
    """
    battery_mw = kwargs["battery_mw"]
    duration_h = kwargs.get("battery_duration_h", 4.0)
    capacity = battery_mw * duration_h
    tolerance = max(capacity * CYCLIC_SOC_TOL_FRACTION, 1e-12)

    relaxed = max(capacity * CYCLIC_SOC_RELAXED_TOL_FRACTION, tolerance)
    evaluations = 0
    history: list[dict] = []
    best: tuple[float, float, DispatchResult] | None = None   # (|gap|, soc0, result)

    def run(soc_mwh: float):
        nonlocal evaluations, best
        soc_pct = 100.0 * soc_mwh / capacity if capacity > 0 else 0.0
        result = simulate(site, controller, initial_soc_pct=soc_pct, **kwargs)
        evaluations += 1
        start = result.metrics["soc_start_mwh"]
        gap = result.metrics["soc_end_mwh"] - start
        history.append({"start_soc_mwh": start, "end_soc_mwh": result.metrics["soc_end_mwh"],
                        "gap_mwh": gap})
        if best is None or abs(gap) < best[0]:
            best = (abs(gap), start, result)
        return gap, result

    def finish(phase: str, gap: float, result: DispatchResult) -> DispatchResult:
        result.metadata["initial_soc_mode"] = "cyclic"
        result.metadata["cyclic_soc"] = {
            "converged": True,
            "phase": phase,
            "evaluations": evaluations,
            "tolerance_mwh": tolerance,
            "relaxed_tolerance_mwh": relaxed,
            "final_gap_mwh": gap,
            "seed_soc_pct": initial_soc_pct,
            "history": history,
        }
        result.metrics["cyclic_soc_residual_mwh"] = float(gap)
        return result

    # ---- phase 1: fixed-point iteration --------------------------------
    soc = (initial_soc_pct / 100.0) * capacity
    for _ in range(max_iterations):
        gap, result = run(soc)
        if abs(gap) <= tolerance:
            return finish("fixed_point", gap, result)
        soc = result.metrics["soc_end_mwh"]

    # ---- phase 2: bisect the residual ----------------------------------
    # end(s) - s is >= 0 at an empty battery and <= 0 at a full one, so a sign
    # change is bracketed by construction even when the map jumps in between.
    low, gap_low = 0.0, run(0.0)[0]
    if abs(gap_low) <= relaxed:
        return finish("bisection", gap_low, best[2])          # type: ignore[index]
    high, gap_high = capacity, run(capacity)[0]
    if abs(gap_high) <= relaxed:
        return finish("bisection", gap_high, best[2])         # type: ignore[index]

    if gap_low > 0 and gap_high < 0:
        for _ in range(40):
            mid = 0.5 * (low + high)
            gap, result = run(mid)
            if abs(gap) <= relaxed:
                return finish("bisection", gap, result)
            if gap > 0:
                low = mid
            else:
                high = mid
            if high - low <= tolerance:
                break

    assert best is not None
    best_gap, _, best_result = best
    if best_gap <= relaxed:
        return finish("best_effort", best_gap if best_gap == 0 else
                      next(h["gap_mwh"] for h in history if abs(h["gap_mwh"]) == best_gap),
                      best_result)

    raise RuntimeError(
        f"Cyclic SOC did not converge after {evaluations} evaluations; best residual "
        f"{best_gap:.6f} MWh against relaxed tolerance {relaxed:.3e} MWh "
        f"(capacity {capacity:.1f} MWh). History: {history}"
    )
