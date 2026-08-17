"""Experiment A: fixed infrastructure, varying control policy.

Builds the standard strategy set for a given site and sizing, runs each through
the *same* dispatcher under the *same* cyclic-SOC boundary condition, and
returns comparable results.

Every strategy sees identical weather, identical hardware, identical demand and
identical starting conditions. The only difference is who decides GPU power.
That is the entire experimental design, and this module is where it is enforced
-- if a strategy needs special handling, it happens here and is visible, rather
than being scattered through call sites.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .control import ComputeController, FixedLoadController, SimpleThrottleController
from .dispatch import DEFAULT_COOLING_FIXED_FRACTION, DispatchResult, simulate_cyclic
from .forecast import NoisySolarForecast, PerfectSolarForecast, SolarForecast
from .gpu import DEFAULT_AGGREGATION, FleetAggregation, PowerPerformanceCurve, get_curve
from .mpc import (
    AnnualPerfectForesightPlanner,
    ForecastMPCController,
    PerfectForesightMPCController,
    ScheduleController,
    build_plant_model,
)
from .scenario import Site

#: Order matters: it is the order results are reported in, and it runs from
#: "no control" through "naive control" to "the theoretical ceiling".
STRATEGY_ORDER = (
    "fixed_load",
    "simple_throttle",
    "perfect_foresight_mpc",
    "perfect_foresight_annual",
)

#: ``forecast_mpc`` is deliberately *not* in :data:`STRATEGY_ORDER`. It is the
#: only strategy with a free parameter that changes the answer -- how wrong the
#: forecast is -- so it cannot have a canonical row in a comparison table the
#: way the others do. It must be asked for, with its error level stated.
KNOWN_STRATEGIES = STRATEGY_ORDER + ("forecast_mpc",)

#: Default belief for ``forecast_mpc``: a plausible day-ahead forecast at
#: default error (ASSUMPTIONS B13). Every result must report the error level it
#: used; this default exists so that "the realistic controller" has one obvious
#: meaning, not so that the level can go unmentioned.
def default_forecast_factory(actual_dc_mw: np.ndarray) -> SolarForecast:
    return NoisySolarForecast(actual_dc_mw)


@dataclass
class Sizing:
    solar_mw: float
    battery_mw: float
    duration_h: float = 4.0

    @property
    def battery_mwh(self) -> float:
        return self.battery_mw * self.duration_h

    def scaled(self, factor: float) -> "Sizing":
        return Sizing(self.solar_mw * factor, self.battery_mw * factor, self.duration_h)

    def as_dict(self) -> dict:
        return {
            "solar_mw_dc": self.solar_mw,
            "battery_mw": self.battery_mw,
            "battery_mwh": self.battery_mwh,
            "battery_duration_h": self.duration_h,
        }


def run_strategy(
    site: Site,
    strategy: str,
    sizing: Sizing,
    *,
    curve: Optional[PowerPerformanceCurve] = None,
    aggregation: FleetAggregation = DEFAULT_AGGREGATION,
    cooling_fixed_fraction: float = DEFAULT_COOLING_FIXED_FRACTION,
    mpc_horizon_hours: int = 48,
    terminal_value_scale: float = 0.95,
    forecast_factory: Optional[Callable[[np.ndarray], SolarForecast]] = None,
) -> DispatchResult:
    """Run one strategy at one sizing, under the cyclic-SOC boundary condition.

    ``forecast_factory`` is consulted only by ``forecast_mpc``; it is handed the
    realised DC solar series and returns the belief the controller plans
    against. Passing it as a factory rather than as loose noise parameters keeps
    the error model's knobs in one place and out of this signature.
    """
    curve = curve or get_curve()
    common = dict(
        solar_mw=sizing.solar_mw,
        battery_mw=sizing.battery_mw,
        battery_duration_h=sizing.duration_h,
        curve=curve,
        aggregation=aggregation,
        cooling_fixed_fraction=cooling_fixed_fraction,
    )
    plant_kwargs = dict(
        solar_mw=sizing.solar_mw, battery_mw=sizing.battery_mw,
        battery_duration_h=sizing.duration_h, curve=curve,
        aggregation=aggregation, cooling_fixed_fraction=cooling_fixed_fraction,
    )
    started = time.time()

    if strategy == "fixed_load":
        controller: ComputeController = FixedLoadController()
        result = simulate_cyclic(site, controller, **common)

    elif strategy == "simple_throttle":
        result = simulate_cyclic(site, SimpleThrottleController(), **common)

    elif strategy == "perfect_foresight_mpc":
        model, fleet = build_plant_model(site, **plant_kwargs)
        controller = PerfectForesightMPCController(
            model=model,
            fleet=fleet,
            forecast=PerfectSolarForecast(model.solar_dc_mw),
            horizon_hours=mpc_horizon_hours,
            terminal_value_scale=terminal_value_scale,
        )
        result = simulate_cyclic(site, controller, **common)

    elif strategy == "forecast_mpc":
        # Built against the *realised* series, exactly as the perfect-foresight
        # path is, and then overwritten hour by hour with the belief. Seeding
        # the plant model with the belief instead would quietly change the
        # inverter cap and the terminal value too, and the point of this
        # comparison is that only the forecast differs.
        model, fleet = build_plant_model(site, **plant_kwargs)
        forecast = (forecast_factory or default_forecast_factory)(model.solar_dc_mw)
        controller = ForecastMPCController(
            model=model,
            fleet=fleet,
            forecast=forecast,
            horizon_hours=mpc_horizon_hours,
            terminal_value_scale=terminal_value_scale,
        )
        result = simulate_cyclic(site, controller, **common)
        # The realised skill of the forecast travels with the result, so no
        # number from this strategy can be quoted without it.
        error = getattr(forecast, "error_stats", None)
        if error is not None:
            result.metrics["forecast_nrmse_24h_pct_of_capacity"] = error()["lead_24h"][
                "nrmse_pct_of_capacity"
            ]

    elif strategy == "perfect_foresight_annual":
        model, _ = build_plant_model(site, **plant_kwargs)
        planner = AnnualPerfectForesightPlanner(model)
        solution = planner.solve()
        result = simulate_cyclic(site, ScheduleController(solution.it_power_mw), **common)
        # Record the planner's own prediction so any divergence between the LP
        # model and the simulator is visible rather than absorbed.
        result.metadata["planner"] = planner.metadata()
        predicted = float(solution.compute_units.sum())
        result.metrics["planner_predicted_compute_units"] = predicted
        result.metrics["planner_model_gap"] = result.metrics["compute_units"] - predicted

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Known: {KNOWN_STRATEGIES}")

    result.metrics["wall_time_s"] = round(time.time() - started, 2)
    result.metadata["sizing"] = sizing.as_dict()
    return result


def run_experiment_a(
    site: Site,
    sizing: Sizing,
    *,
    strategies: tuple[str, ...] = STRATEGY_ORDER,
    progress: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> dict[str, DispatchResult]:
    """Run every strategy at one fixed sizing."""
    results: dict[str, DispatchResult] = {}
    for strategy in strategies:
        if progress:
            progress(strategy)
        results[strategy] = run_strategy(site, strategy, sizing, **kwargs)
    return results


def derate_sweep(
    site: Site,
    base: Sizing,
    scales: tuple[float, ...],
    *,
    strategies: tuple[str, ...] = STRATEGY_ORDER,
    progress: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> dict[float, dict[str, DispatchResult]]:
    """Shrink the infrastructure and watch what each policy retains.

    The fixed-load-optimal sizing is built for a 99%-uptime tail, so it has
    little to teach about compute. The interesting regime -- and the one
    Experiment B will search -- is under-built infrastructure, where stored
    energy actually binds.
    """
    out: dict[float, dict[str, DispatchResult]] = {}
    for scale in scales:
        if progress:
            progress(f"scale {scale:.2f}")
        out[scale] = run_experiment_a(
            site, base.scaled(scale), strategies=strategies, progress=None, **kwargs
        )
    return out


def infrastructure_for_target_compute(
    sweep: dict[float, dict[str, DispatchResult]], strategy: str, target_compute: float
) -> Optional[float]:
    """Smallest scale factor reaching ``target_compute``, by linear interpolation.

    A first look at the Experiment B question -- *how much less infrastructure
    for the same work?* -- read off an Experiment A sweep. It interpolates
    between sampled scales rather than optimising, so treat it as an indication
    of magnitude, not the final answer.
    """
    points = sorted(
        (scale, runs[strategy].metrics["compute_units"])
        for scale, runs in sweep.items()
        if strategy in runs
    )
    if not points or points[-1][1] < target_compute:
        return None
    for (s0, c0), (s1, c1) in zip(points, points[1:]):
        if c0 <= target_compute <= c1:
            if c1 == c0:
                return s0
            return s0 + (s1 - s0) * (target_compute - c0) / (c1 - c0)
    return points[0][0] if points[0][1] >= target_compute else None


# ---------------------------------------------------------------------------
# Experiment B: equal compute, minimum capital
# ---------------------------------------------------------------------------

@dataclass
class SizingSearchResult:
    strategy: str
    sizing: Sizing
    capex_musd: float
    compute_units: float
    target_compute: float
    involuntary_shortfall_mwh: float
    breakdown: dict
    evaluations: int
    wall_time_s: float
    success: bool
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "sizing": self.sizing.as_dict(),
            "capex_musd": self.capex_musd,
            "compute_units": self.compute_units,
            "target_compute": self.target_compute,
            "involuntary_shortfall_mwh": self.involuntary_shortfall_mwh,
            "capex_breakdown": self.breakdown,
            "evaluations": self.evaluations,
            "wall_time_s": self.wall_time_s,
            "success": self.success,
            "metrics": self.metrics,
        }


def minimum_capex_for_compute(
    site: Site,
    strategy: str,
    *,
    target_compute: float,
    cost_model,
    reference: Sizing,
    solar_bounds: tuple[float, float] = (0.15, 1.60),
    battery_power_bounds: tuple[float, float] = (0.05, 1.60),
    duration_bounds: tuple[float, float] = (0.5, 16.0),
    seed: int = 20260815,
    maxiter: int = 20,
    popsize: int = 10,
    progress: Optional[Callable[[str], None]] = None,
    **run_kwargs,
) -> SizingSearchResult:
    """Cheapest solar / battery-power / battery-energy design meeting a compute target.

    This is the Experiment B question. Battery power and energy are searched
    independently -- the whole point of decoupling them (ASSUMPTIONS Q2) -- with
    duration as the third decision variable so the box stays rectangular.

    The constraint enters as a penalty rather than a hard filter because a
    rejected design gives the optimiser no gradient to follow; a design 1% short
    on compute is charged far more than any plausible capex, so the optimum sits
    on the constraint.

    Bounds are expressed as multiples of ``reference`` (the fixed-load-optimal
    plant) so the search box scales with the site rather than being hard-coded.
    """
    from scipy.optimize import differential_evolution

    started = time.time()
    cache: dict[tuple, tuple[float, float, float, dict]] = {}
    # Steep enough that a 0.01% compute shortfall costs more than any capex the
    # optimiser could save by under-building.
    penalty_musd = 1.0e6

    def evaluate(solar_mw: float, battery_mw: float, duration_h: float):
        # Round to a grid so the optimiser's fine probing does not re-run
        # near-identical (and expensive) simulations.
        key = (round(solar_mw, 2), round(battery_mw, 2), round(duration_h, 3))
        if key in cache:
            return cache[key]
        sizing = Sizing(key[0], key[1], key[2])
        run = run_strategy(site, strategy, sizing, **run_kwargs)
        capex = cost_model.total_capex_usd(
            sizing.solar_mw, sizing.battery_mw, sizing.battery_mwh
        ) / 1e6
        out = (capex, run.metrics["compute_units"],
               run.metrics["involuntary_shortfall_mwh"], run.metrics)
        cache[key] = out
        return out

    def objective(x) -> float:
        solar_mult, battery_mult, duration_h = x
        capex, compute, _, _ = evaluate(
            solar_mult * reference.solar_mw, battery_mult * reference.battery_mw, duration_h
        )
        shortfall = max(0.0, (target_compute - compute) / target_compute)
        return capex + penalty_musd * shortfall

    if progress:
        progress(f"{strategy}: searching for target {target_compute:,.1f}")

    result = differential_evolution(
        objective,
        [solar_bounds, battery_power_bounds, duration_bounds],
        maxiter=maxiter, popsize=popsize, seed=seed, polish=False, disp=False,
    )

    solar_mult, battery_mult, duration_h = result.x
    sizing = Sizing(
        round(solar_mult * reference.solar_mw, 2),
        round(battery_mult * reference.battery_mw, 2),
        round(duration_h, 3),
    )
    capex, compute, shortfall, metrics = evaluate(sizing.solar_mw, sizing.battery_mw, sizing.duration_h)

    # Repair: the penalty steers the search onto the constraint but the grid
    # rounding can still leave the winner a hair short, and a design that misses
    # the target is not an answer to the question. Grow it uniformly until it
    # qualifies, then bisect back to the smallest factor that still does.
    if compute < target_compute:
        low, high = 1.0, 1.0
        for _ in range(8):
            high *= 1.05
            grown = Sizing(round(sizing.solar_mw * high, 2),
                           round(sizing.battery_mw * high, 2), sizing.duration_h)
            capex, compute, shortfall, metrics = evaluate(
                grown.solar_mw, grown.battery_mw, grown.duration_h)
            if compute >= target_compute:
                sizing = grown
                break
            low = high
        for _ in range(10):
            mid = 0.5 * (low + high)
            candidate = Sizing(round(sizing.solar_mw / high * mid, 2),
                               round(sizing.battery_mw / high * mid, 2), sizing.duration_h)
            c_capex, c_compute, c_short, c_metrics = evaluate(
                candidate.solar_mw, candidate.battery_mw, candidate.duration_h)
            if c_compute >= target_compute:
                high, sizing = mid, candidate
                capex, compute, shortfall, metrics = c_capex, c_compute, c_short, c_metrics
            else:
                low = mid

    return SizingSearchResult(
        strategy=strategy,
        sizing=sizing,
        capex_musd=capex,
        compute_units=compute,
        target_compute=target_compute,
        involuntary_shortfall_mwh=shortfall,
        breakdown=cost_model.breakdown(sizing.solar_mw, sizing.battery_mw, sizing.battery_mwh),
        evaluations=len(cache),
        wall_time_s=round(time.time() - started, 1),
        success=compute >= target_compute * (1 - 1e-6),
        metrics=metrics,
    )
