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

from .control import (
    CaseyGovernor,
    ComputeController,
    FixedLoadController,
    SimpleThrottleController,
)
from .solar_clock import build_solar_clock
from .dispatch import DEFAULT_COOLING_FIXED_FRACTION, DispatchResult, simulate_cyclic
from .forecast import (
    REFERENCE_NRMSE_24H_PCT,
    NoisySolarForecast,
    PerfectSolarForecast,
    SolarForecast,
    forecast_at_realised_nrmse,
)
from .gpu import DEFAULT_AGGREGATION, FleetAggregation, PowerPerformanceCurve, get_curve
from .mpc import (
    AnnualPerfectForesightPlanner,
    ForecastMPCController,
    PerfectForesightMPCController,
    ScheduleController,
    build_plant_model,
)
from .scenario import Site

#: The comparison ladder. Order matters: it is the order results are reported
#: in, and it climbs from "no control" through two forecast-free heuristics to
#: the theoretical ceiling.
#:
#: The rungs differ in *what the controller is allowed to know*, which is the
#: only variable that matters here:
#:
#:   fixed_load               nothing; demand is exogenous
#:   simple_throttle          battery SOC
#:   casey_governor           SOC + clock + present generation (no forecast)
#:   perfect_foresight_mpc    the realised future, over a finite horizon
#:   perfect_foresight_annual the realised future, over the whole year
#:
#: ``casey_governor`` is the last rung a real operator could build without a
#: weather forecast, so the step from it to the MPC rungs is the value of
#: forecasting, and the step from ``fixed_load`` to it is the value of merely
#: reacting to stored energy.
STRATEGY_ORDER = (
    "fixed_load",
    "simple_throttle",
    "casey_governor",
    "perfect_foresight_mpc",
    "perfect_foresight_annual",
)

#: ``forecast_mpc`` is deliberately *not* in :data:`STRATEGY_ORDER`. It is the
#: only strategy with a free parameter that changes the answer -- how wrong the
#: forecast is -- so it cannot have a canonical row in a comparison table the
#: way the others do. It must be asked for, with its error level stated.
KNOWN_STRATEGIES = STRATEGY_ORDER + ("forecast_mpc",)

#: The full ladder, including the one deployable forecast-aware design, pinned
#: at the reference skill level.
#:
#: This is a *different* object from :data:`STRATEGY_ORDER` rather than an
#: extension of it, because including ``forecast_mpc`` is only defensible now
#: that its error is specified in **realised day-ahead nRMSE** rather than in
#: the model's internal sigma. Calibrating to realised error is what makes "10%"
#: mean the same thing in every weather year; the same sigma lands anywhere
#: between 9.5% and 10.5% depending on the year's cloud envelope. Any table
#: built from this ladder must still print the realised error achieved.
COMPARISON_LADDER = (
    "fixed_load",
    "simple_throttle",
    "casey_governor",
    "forecast_mpc",
    "perfect_foresight_mpc",
    "perfect_foresight_annual",
)

#: Default belief for ``forecast_mpc``: a day-ahead forecast calibrated to the
#: reference *realised* skill level (ASSUMPTIONS B13). Every result must report
#: the error it actually achieved; this default exists so that "the realistic
#: controller" has one obvious meaning, not so the level can go unmentioned.
def default_forecast_factory(actual_dc_mw: np.ndarray) -> SolarForecast:
    return forecast_at_realised_nrmse(actual_dc_mw, REFERENCE_NRMSE_24H_PCT)


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
    casey_solar_return_fraction: float = 1.0,
    casey_reserve_fraction: float = 0.0,
    forecast_nrmse_pct: float = REFERENCE_NRMSE_24H_PCT,
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

    elif strategy == "casey_governor":
        # The governor gets plant *scalars* and an *ephemeris*, and that is the
        # whole of its knowledge. Note what is deliberately not passed: the
        # PlantModel itself (which carries the realised solar series) and any
        # SolarForecast. Built through build_plant_model only to reuse the
        # multiplier arithmetic; .constants() then discards every time series.
        model, _ = build_plant_model(site, **plant_kwargs)
        clock = build_solar_clock(site)
        controller = CaseyGovernor(
            plant=model.constants(cooling_fixed_fraction=cooling_fixed_fraction),
            hours_to_sunrise=clock.hours_to_sunrise,
            solar_return_fraction=casey_solar_return_fraction,
            reserve_fraction=casey_reserve_fraction,
        )
        result = simulate_cyclic(site, controller, **common)
        result.metadata["solar_clock"] = clock.metadata()

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
        if forecast_factory is not None:
            forecast = forecast_factory(model.solar_dc_mw)
        else:
            # Calibrated to a *realised* error target, not to a sigma, so the
            # same nominal skill level means the same thing in every year.
            forecast = forecast_at_realised_nrmse(model.solar_dc_mw, forecast_nrmse_pct)
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
            stats = error()
            result.metrics["forecast_nrmse_24h_pct_of_capacity"] = stats["lead_24h"][
                "nrmse_pct_of_capacity"
            ]
            result.metrics["forecast_nrmse_24h_pct_of_mean_output"] = stats["lead_24h"][
                "nrmse_pct_of_mean_output"
            ]
            result.metrics["forecast_target_nrmse_pct"] = float(forecast_nrmse_pct)
            result.metrics["forecast_sigma_24h"] = float(
                getattr(forecast, "sigma_24h", float("nan"))
            )

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

#: Durations the battery cost decomposition's source data actually spans
#: (NLR/NREL publish 2/4/6/8/10-hour systems). Beyond this the affine
#: power/energy split is an extrapolation and must be labelled as one.
COST_SOURCE_DURATION_RANGE_H = (2.0, 10.0)


def duration_is_extrapolated(duration_h: float) -> bool:
    low, high = COST_SOURCE_DURATION_RANGE_H
    return not (low <= duration_h <= high)


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
    #: Reliability cap this search was run under, if any (B2). ``None`` is B1.
    max_shortfall_mwh: Optional[float] = None
    reliability_met: bool = True
    duration_mode: str = "free"

    @property
    def duration_extrapolated(self) -> bool:
        """True when the chosen duration sits outside the cost source's range."""
        return duration_is_extrapolated(self.sizing.duration_h)

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
            "max_shortfall_mwh": self.max_shortfall_mwh,
            "reliability_met": self.reliability_met,
            "duration_mode": self.duration_mode,
            "duration_extrapolated": self.duration_extrapolated,
            "cost_source_duration_range_h": list(COST_SOURCE_DURATION_RANGE_H),
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
    fixed_duration_h: Optional[float] = None,
    max_shortfall_mwh: Optional[float] = None,
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

    Constraints enter as penalties rather than hard filters because a rejected
    design gives the optimiser no gradient to follow. Both penalties are steep
    enough that the optimum sits on the constraint rather than trading against
    it.

    ``max_shortfall_mwh`` is the **B2 reliability constraint**. Without it (B1)
    a design may reach the compute target by browning out, and comparing a
    fixed-load plant that books thousands of MWh of involuntary shortfall
    against a flexible one that books none is not a like-for-like capital
    comparison. The cap is an absolute annual energy figure applied identically
    to every strategy, which is deliberately *not* a criterion that favours
    flexible operation: a flexible controller has to buy its own reliability out
    of the same capital budget, and a fixed-load plant that was already reliable
    pays nothing for the constraint.

    ``fixed_duration_h`` pins battery duration and searches the remaining two
    variables, for the duration sensitivity (Task 7). Note that the cost model's
    power/energy split is sourced over 2-10 hours; outside that range the result
    is an extrapolation and ``SizingSearchResult.duration_extrapolated`` says so.

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

    def feasible(compute: float, shortfall: float) -> bool:
        if compute < target_compute * (1 - 1e-6):
            return False
        return max_shortfall_mwh is None or shortfall <= max_shortfall_mwh

    def violation(compute: float, shortfall: float) -> float:
        """Combined relative constraint violation; zero when feasible."""
        short_on_compute = max(0.0, (target_compute - compute) / target_compute)
        if max_shortfall_mwh is None:
            return short_on_compute
        # Normalised by the cap itself so the two penalties are commensurate.
        scale = max(max_shortfall_mwh, 1.0)
        excess = max(0.0, (shortfall - max_shortfall_mwh) / scale)
        return short_on_compute + excess

    def unpack(x) -> tuple[float, float, float]:
        if fixed_duration_h is None:
            solar_mult, battery_mult, duration_h = x
        else:
            solar_mult, battery_mult = x
            duration_h = fixed_duration_h
        return (
            solar_mult * reference.solar_mw,
            battery_mult * reference.battery_mw,
            duration_h,
        )

    def objective(x) -> float:
        capex, compute, shortfall, _ = evaluate(*unpack(x))
        value = capex + penalty_musd * violation(compute, shortfall)
        # A non-finite objective poisons the whole population: differential
        # evolution takes min/mean over it, and one NaN propagates to every
        # convergence test. Map it to a cost worse than any real design instead,
        # so such a point is simply rejected. Nothing in the physics is expected
        # to produce one -- the search-box extremes were probed and do not --
        # but the failure mode is silent and the guard costs nothing.
        if not np.isfinite(value):
            return float(penalty_musd * 10.0)
        return value

    if progress:
        progress(f"{strategy}: searching for target {target_compute:,.1f}")

    bounds = [solar_bounds, battery_power_bounds]
    if fixed_duration_h is None:
        bounds.append(duration_bounds)

    # Deliberately serial. Two reasons: ``objective`` is a closure and cannot be
    # pickled to worker processes, and -- more importantly -- the memo cache
    # above is what makes this search affordable at all, and a process pool
    # would give every worker its own empty copy of it. Parallelism belongs one
    # level up, across independent searches; see ``search_sizing`` below.
    result = differential_evolution(
        objective, bounds,
        maxiter=maxiter, popsize=popsize, seed=seed, polish=False, disp=False,
    )

    solar_mw, battery_mw, duration_h = unpack(result.x)
    sizing = Sizing(round(solar_mw, 2), round(battery_mw, 2), round(duration_h, 3))
    capex, compute, shortfall, metrics = evaluate(
        sizing.solar_mw, sizing.battery_mw, sizing.duration_h
    )

    # Repair: the penalty steers the search onto the constraint but the grid
    # rounding can still leave the winner a hair short, and a design that misses
    # the target is not an answer to the question. Grow it uniformly until it
    # qualifies, then bisect back to the smallest factor that still does.
    # Growing helps both constraints: more plant means more compute *and* less
    # shortfall, so the same repair serves B1 and B2.
    if not feasible(compute, shortfall):
        low, high = 1.0, 1.0
        for _ in range(10):
            high *= 1.05
            grown = Sizing(round(sizing.solar_mw * high, 2),
                           round(sizing.battery_mw * high, 2), sizing.duration_h)
            capex, compute, shortfall, metrics = evaluate(
                grown.solar_mw, grown.battery_mw, grown.duration_h)
            if feasible(compute, shortfall):
                sizing = grown
                break
            low = high
        for _ in range(10):
            mid = 0.5 * (low + high)
            candidate = Sizing(round(sizing.solar_mw / high * mid, 2),
                               round(sizing.battery_mw / high * mid, 2), sizing.duration_h)
            c_capex, c_compute, c_short, c_metrics = evaluate(
                candidate.solar_mw, candidate.battery_mw, candidate.duration_h)
            if feasible(c_compute, c_short):
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
        success=feasible(compute, shortfall),
        metrics=metrics,
        max_shortfall_mwh=max_shortfall_mwh,
        reliability_met=(max_shortfall_mwh is None or shortfall <= max_shortfall_mwh),
        duration_mode="free" if fixed_duration_h is None else f"fixed_{fixed_duration_h:g}h",
    )


# ---------------------------------------------------------------------------
# Running many sizing searches in parallel
# ---------------------------------------------------------------------------
#
# One sizing search is inherently serial: it is a population search whose
# affordability depends entirely on memoising repeated designs, and splitting it
# across processes would hand each worker an empty cache. Independent searches,
# on the other hand -- different strategies, durations, variants, years -- share
# nothing, so that is the level at which to parallelise.


@dataclass(frozen=True)
class SizingSearchSpec:
    """One complete, picklable description of a sizing search.

    Carries a :class:`~flexcompute.scenario.Scenario` rather than a ``Site`` so
    it can cross a process boundary; the worker rebuilds the site (~0.7 s),
    which is cheaper and far less fragile than pickling live upstream objects.
    """

    scenario: "object"                 # Scenario
    strategy: str
    target_compute: float
    reference: Sizing
    label: str = ""
    fixed_duration_h: Optional[float] = None
    max_shortfall_mwh: Optional[float] = None
    seed: int = 20260815
    maxiter: int = 20
    popsize: int = 10
    curve_name: Optional[str] = None
    aggregation: FleetAggregation = DEFAULT_AGGREGATION
    cooling_fixed_fraction: float = DEFAULT_COOLING_FIXED_FRACTION
    forecast_nrmse_pct: float = REFERENCE_NRMSE_24H_PCT


def search_sizing(spec: SizingSearchSpec) -> tuple[str, SizingSearchResult]:
    """Worker entry point for one sizing search. Importable for pickling."""
    import logging

    from .costs import from_upstream_config

    logging.disable(logging.INFO)
    site = spec.scenario.build()
    cost_model = from_upstream_config(site.config, spec.scenario.architecture)
    result = minimum_capex_for_compute(
        site,
        spec.strategy,
        target_compute=spec.target_compute,
        cost_model=cost_model,
        reference=spec.reference,
        fixed_duration_h=spec.fixed_duration_h,
        max_shortfall_mwh=spec.max_shortfall_mwh,
        seed=spec.seed,
        maxiter=spec.maxiter,
        popsize=spec.popsize,
        curve=get_curve(spec.curve_name) if spec.curve_name else get_curve(),
        aggregation=spec.aggregation,
        cooling_fixed_fraction=spec.cooling_fixed_fraction,
        forecast_nrmse_pct=spec.forecast_nrmse_pct,
    )
    return spec.label, result


def run_sizing_searches(
    specs: "list[SizingSearchSpec]",
    *,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int, str, "SizingSearchResult"], None]] = None,
) -> "dict[str, SizingSearchResult]":
    """Execute independent sizing searches in parallel.

    ``progress`` receives the finished result along with the counters, so a
    caller can report each search as it lands rather than waiting for the whole
    batch. Searches complete out of order.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .multiyear import default_workers

    workers = default_workers() if workers is None else workers
    out: dict[str, SizingSearchResult] = {}

    if workers <= 1:
        for i, spec in enumerate(specs, 1):
            label, result = search_sizing(spec)
            out[label] = result
            if progress:
                progress(i, len(specs), label, result)
        return out

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(search_sizing, spec) for spec in specs]
        for i, future in enumerate(as_completed(futures), 1):
            label, result = future.result()
            out[label] = result
            if progress:
                progress(i, len(specs), label, result)
    return out
