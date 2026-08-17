"""Model-predictive control: plan a GPU power schedule by optimisation.

Why this is a linear program
----------------------------
The fleet's compute-vs-power curve is **concave** once fleet aggregation is
applied (see :meth:`PowerPerformanceCurve.concave_hull`), and every other
relationship in the dispatch model is linear in GPU power:

* bus load is exactly ``u * alpha`` with ``alpha = m_it + (pue - 1) * m_cool``;
* battery state is a linear recursion in charge and discharge;
* conversion losses are constant multipliers.

Maximising a concave piecewise-linear objective over linear constraints is an
LP. Written as ``compute <= a_i * x + b_i`` for each hull segment, the solver
finds the global optimum exactly -- no heuristics, no local minima, nothing to
tune. That transparency is the point: when the MPC beats a heuristic we can say
precisely why, because the optimum is the optimum.

Simultaneous charge and discharge are left unconstrained rather than forced
apart with a binary, because round-trip efficiency is strictly below one, so
doing both in the same hour always loses energy and is never optimal. Verified
by test.

Three planners, and only one of them is buildable
------------------------------------------------
:class:`AnnualPerfectForesightPlanner`
    One LP over all 8760 hours with a cyclic SOC constraint. This is the true
    perfect-foresight ceiling: no horizon truncation, no terminal-value
    guesswork, no free starting energy. **Upper bound.**

:class:`PerfectForesightMPCController`
    Receding horizon, re-solved every hour, executing only the first action.
    Same physics, same objective, but finite lookahead and therefore a terminal
    value function. The gap to the annual planner *is* the cost of a finite
    horizon. **Upper bound.**

:class:`ForecastMPCController`
    The same receding-horizon machinery given a *belief* rather than the truth.
    With :class:`~flexcompute.forecast.NoisySolarForecast` this is a design
    somebody could actually deploy, and the gap to the perfect-foresight MPC is
    the cost of not knowing -- which is the number that decides whether any of
    the capital savings in this project survive contact with reality.

The first two consume the realised future and exist to bound what
forecast-aware control could be worth. The third has to earn it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .control import Observation
from .forecast import PerfectSolarForecast, SolarForecast
from .dispatch import DEFAULT_COOLING_FIXED_FRACTION
from .gpu import DEFAULT_AGGREGATION, FleetAggregation, GpuFleet, PowerPerformanceCurve, get_curve
from .scenario import Site
from .upstream_bridge import ensure_upstream_importable

HOURS_PER_YEAR = 8760


# ---------------------------------------------------------------------------
# The physical model the planner optimises over
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlantModel:
    """Everything the LP needs about the plant, precomputed and linear.

    Deliberately a plain data object: the same model is handed to the annual
    planner and to the receding-horizon controller, so the two cannot silently
    optimise against different physics.
    """

    alpha: np.ndarray            # bus MW per IT MW, per hour
    beta: np.ndarray             # bus MW drawn regardless of IT power (fixed cooling)
    solar_dc_mw: np.ndarray      # available PV DC, per hour
    demand_mw: np.ndarray        # unconstrained demand, per hour
    m_solar_bus: float
    m_solar_batt: float
    m_batt_bus: float
    battery_mw: float
    battery_mwh: float
    min_power_fraction: float
    hull_slopes: np.ndarray      # a_i for compute <= a_i * x + b_i
    hull_intercepts: np.ndarray  # b_i
    inverter_cap_mw_dc: Optional[float] = None

    @property
    def horizon(self) -> int:
        return len(self.alpha)

    @property
    def solar_to_bus_max(self) -> np.ndarray:
        """Most PV that can reach the bus, at the bus, per hour.

        Derived rather than stored so that swapping in a *believed* solar series
        re-applies the inverter cap to that series. Storing it as a ratio
        against the realised series would silently mis-cap any forecast that
        differs from the truth -- correct today only because perfect foresight
        makes them equal.
        """
        reachable = (
            self.solar_dc_mw if self.inverter_cap_mw_dc is None
            else np.minimum(self.solar_dc_mw, self.inverter_cap_mw_dc)
        )
        return reachable / self.m_solar_bus

    def with_solar(self, solar_dc_mw: np.ndarray) -> "PlantModel":
        """Same plant, different belief about PV availability."""
        believed = np.asarray(solar_dc_mw, dtype=float)
        if believed.shape != self.solar_dc_mw.shape:
            raise ValueError(
                f"solar series must have {self.solar_dc_mw.shape} entries, got {believed.shape}"
            )
        return replace(self, solar_dc_mw=believed)

    def marginal_compute_per_battery_mwh(self) -> float:
        """Compute obtainable from one stored MWh, spent at peak efficiency.

        Used as the terminal value of stored energy in the receding-horizon
        controller. It is an *upper* bound on what the energy is really worth
        -- it assumes the energy is spent at the most compute-efficient
        operating point and never curtailed -- which makes the controller
        appropriately reluctant to drain the battery near its horizon edge.
        """
        hull_x, hull_y = _hull_points(
            self.hull_slopes, self.hull_intercepts, self.min_power_fraction
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            efficiency = np.where(hull_x > 0, hull_y / np.where(hull_x > 0, hull_x, 1.0), 0.0)
        best = float(efficiency.max())            # compute-fraction per power-fraction

        demand = float(np.mean(self.demand_mw))
        alpha = float(np.mean(self.alpha))
        if demand <= 0 or alpha <= 0:
            return 0.0
        # battery MWh -> bus MWh -> IT MWh -> compute units
        it_mwh_per_battery_mwh = (1.0 / self.m_batt_bus) / alpha
        compute_per_it_mwh = best / demand
        return it_mwh_per_battery_mwh * compute_per_it_mwh

    def window(self, start: int, hours: int) -> "PlantModel":
        """Slice a horizon out of the annual model, **truncated** at year end.

        Truncation rather than padding, deliberately. Padding the tail with
        zero solar invents a week-long night every December and drives the
        planner into artificial panic -- it was observed making the LP outright
        infeasible at long horizons. Truncation says the honest thing instead:
        near the end of the modelled year there is simply less future to plan
        over, and the terminal value carries what is left.

        Wrapping around to January would be worse still: that is not foresight,
        it is acausality.
        """
        end = min(start + hours, len(self.alpha))

        def take(arr: np.ndarray) -> np.ndarray:
            return arr[start:end]

        return replace(
            self,
            alpha=take(self.alpha),
            beta=take(self.beta),
            solar_dc_mw=take(self.solar_dc_mw),
            demand_mw=take(self.demand_mw),
        )


def _hull_points(
    slopes: np.ndarray, intercepts: np.ndarray, x_min: float
) -> tuple[np.ndarray, np.ndarray]:
    """Recover hull vertices from the affine pieces (for the efficiency scan).

    Anchored at ``x_min`` -- the fleet's lowest operating point -- rather than
    at zero, because the affine pieces extrapolate to negative compute below
    the hull's domain and that is not a point the fleet can occupy.
    """
    xs = [x_min]
    for i in range(len(slopes) - 1):
        # intersection of consecutive affine pieces
        xs.append((intercepts[i + 1] - intercepts[i]) / (slopes[i] - slopes[i + 1]))
    xs.append(1.0)
    x = np.array(xs)
    y = np.min(np.stack([slopes[i] * x + intercepts[i] for i in range(len(slopes))]), axis=0)
    return x, np.clip(y, 0.0, None)


def build_plant_model(
    site: Site,
    *,
    solar_mw: float,
    battery_mw: float,
    battery_duration_h: float = 4.0,
    curve: Optional[PowerPerformanceCurve] = None,
    aggregation: FleetAggregation = DEFAULT_AGGREGATION,
    cooling_fixed_fraction: float = DEFAULT_COOLING_FIXED_FRACTION,
    forecast: Optional[SolarForecast] = None,
    unconstrained_demand_mw: Optional[np.ndarray] = None,
) -> tuple[PlantModel, GpuFleet]:
    """Assemble the LP's view of the plant from a site and a sizing.

    The solar series comes from ``forecast`` if one is given, so a
    forecast-aware controller plans against its belief rather than the truth.
    With no forecast the realised series is used, which is only appropriate for
    the perfect-foresight bound.
    """
    ensure_upstream_importable()
    from power_systems_estimator import PowerFlowAnalyzer  # type: ignore[import-not-found]

    curve = curve or get_curve()
    fleet = GpuFleet(curve=curve, total_gpus=site.scenario.total_gpus, aggregation=aggregation)

    analyzer = PowerFlowAnalyzer(site.config, topology=site.scenario.topology)
    mult = analyzer.get_bus_architecture_multipliers(site.scenario.architecture)

    demand = (
        site.it_load_mw if unconstrained_demand_mw is None
        else np.asarray(unconstrained_demand_mw, dtype=float)
    )
    pue = site.hourly_pue
    # Bus load is affine in GPU power once cooling has a fixed component:
    #   bus = alpha * u + beta      (ASSUMPTIONS Q3)
    # With cooling_fixed_fraction = 0 this collapses to the purely proportional
    # form and beta vanishes, which is what keeps the baseline reproducible.
    alpha = (
        mult["bus_to_it"]
        + (1.0 - cooling_fixed_fraction) * (pue - 1.0) * mult["bus_to_cooling"]
    )
    beta = cooling_fixed_fraction * demand * (pue - 1.0) * mult["bus_to_cooling"]

    solar_dc_mw = site.solar_p_dc * solar_mw
    inverter_cap = None
    if site.scenario.architecture == "ac_coupled":
        limit = solar_mw / site.config.efficiency.inverter_load_ratio
        if site.scenario.topology == "lv_direct":
            solar_dc_mw = np.minimum(solar_dc_mw, limit)
        else:
            inverter_cap = limit

    believed = (
        solar_dc_mw if forecast is None
        else np.asarray(forecast.horizon(0, len(solar_dc_mw)), dtype=float)
    )

    hull = fleet.effective_curve
    slopes = np.diff(hull.compute_fraction) / np.diff(hull.power_fraction)
    intercepts = hull.compute_fraction[:-1] - slopes * hull.power_fraction[:-1]

    model = PlantModel(
        alpha=alpha,
        beta=beta,
        solar_dc_mw=believed,
        demand_mw=demand,
        m_solar_bus=mult["solar_to_bus"],
        m_solar_batt=mult["solar_to_battery"],
        m_batt_bus=mult["battery_to_bus"],
        battery_mw=battery_mw,
        battery_mwh=battery_mw * battery_duration_h,
        min_power_fraction=fleet.min_power_fraction,
        hull_slopes=slopes,
        hull_intercepts=intercepts,
        inverter_cap_mw_dc=inverter_cap,
    )
    return model, fleet


# ---------------------------------------------------------------------------
# The LP itself
# ---------------------------------------------------------------------------

@dataclass
class ScheduleSolution:
    it_power_mw: np.ndarray
    soc_mwh: np.ndarray
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    compute_units: np.ndarray
    unserved_bus_mwh: np.ndarray
    objective: float
    status: str


def solve_schedule(
    model: PlantModel,
    *,
    initial_soc_mwh: Optional[float],
    terminal_soc_mwh: Optional[float] = None,
    cyclic: bool = False,
    terminal_value_per_mwh: float = 0.0,
    unserved_penalty_per_mwh: Optional[float] = None,
    solver: str = "HIGHS",
) -> ScheduleSolution:
    """Maximise compute over the horizon subject to the plant's physics.

    ``initial_soc_mwh=None`` leaves the starting state free, which combined with
    ``cyclic=True`` lets the annual planner choose the best self-sustaining
    trajectory -- the honest way to grant no free energy while not arbitrarily
    fixing the January state.

    A slack variable represents **unserved bus energy** -- a brownout. Without
    it the LP is simply infeasible whenever the plant cannot cover even a fully
    parked fleet, which is precisely the multi-day drought where control matters
    most; an optimiser that crashes there is useless. The slack is priced far
    above any compute it could buy, so it is a genuine last resort and never a
    trade the planner makes voluntarily.
    """
    import cvxpy as cp

    n = model.horizon
    u = cp.Variable(n, name="it_power_mw")
    solar_to_bus = cp.Variable(n, nonneg=True, name="solar_to_bus_mw")
    charge = cp.Variable(n, nonneg=True, name="charge_mw")
    discharge = cp.Variable(n, nonneg=True, name="discharge_mw")
    compute = cp.Variable(n, name="compute_units")
    unserved = cp.Variable(n, nonneg=True, name="unserved_bus_mwh")
    soc = cp.Variable(n + 1, name="soc_mwh")

    if unserved_penalty_per_mwh is None:
        # One hour of full bus load is worth one compute-unit, so compute per
        # bus MWh is ~1/mean(bus load). Price a shortfall a thousand times
        # higher: effectively lexicographic, while staying well conditioned.
        full_bus_load = float(np.mean(model.demand_mw * model.alpha + model.beta))
        unserved_penalty_per_mwh = 1000.0 / full_bus_load if full_bus_load > 0 else 1e3

    constraints = [
        # GPU power bounds
        u >= model.min_power_fraction * model.demand_mw,
        u <= model.demand_mw,
        # Concave piecewise-linear compute, as the min of its affine pieces
        compute >= 0,
        # Bus balance. The unserved slack is the only way the plan can fall
        # short, and it is priced out of contention.
        solar_to_bus + discharge / model.m_batt_bus + unserved
        == cp.multiply(u, model.alpha) + model.beta,
        # PV DC budget: the load path and the charge path share one array
        solar_to_bus * model.m_solar_bus + charge * model.m_solar_batt <= model.solar_dc_mw,
        solar_to_bus <= model.solar_to_bus_max,
        # Battery ratings and state
        charge <= model.battery_mw,
        discharge <= model.battery_mw,
        soc[1:] == soc[:-1] + charge - discharge,
        soc >= 0,
        soc <= model.battery_mwh,
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        power_fraction = cp.multiply(u, 1.0 / np.where(model.demand_mw > 0, model.demand_mw, 1.0))
    for slope, intercept in zip(model.hull_slopes, model.hull_intercepts):
        constraints.append(compute <= slope * power_fraction + intercept)

    if initial_soc_mwh is not None:
        constraints.append(soc[0] == initial_soc_mwh)
    if cyclic:
        constraints.append(soc[n] >= soc[0])
    if terminal_soc_mwh is not None:
        constraints.append(soc[n] >= terminal_soc_mwh)

    objective = cp.sum(compute) - unserved_penalty_per_mwh * cp.sum(unserved)
    if terminal_value_per_mwh:
        objective = objective + terminal_value_per_mwh * soc[n]

    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=solver)

    if u.value is None:
        raise RuntimeError(f"MPC solve failed with status '{problem.status}'")

    return ScheduleSolution(
        it_power_mw=np.asarray(u.value, dtype=float),
        soc_mwh=np.asarray(soc.value, dtype=float),
        charge_mw=np.asarray(charge.value, dtype=float),
        discharge_mw=np.asarray(discharge.value, dtype=float),
        compute_units=np.asarray(compute.value, dtype=float),
        unserved_bus_mwh=np.asarray(unserved.value, dtype=float),
        objective=float(problem.value),
        status=str(problem.status),
    )


# ---------------------------------------------------------------------------
# Planner and controller
# ---------------------------------------------------------------------------

@dataclass
class AnnualPerfectForesightPlanner:
    """One LP over the whole year. The ceiling.

    No horizon truncation and no terminal-value approximation, so nothing about
    the answer is an artefact of MPC tuning. The cyclic constraint means the
    year must be self-sustaining, so the plan gets no free starting energy
    either (ASSUMPTIONS Q1).
    """

    model: PlantModel
    name: str = "perfect_foresight_annual"
    solution: Optional[ScheduleSolution] = None

    def solve(self, *, solver: str = "HIGHS") -> ScheduleSolution:
        self.solution = solve_schedule(
            self.model, initial_soc_mwh=None, cyclic=True, solver=solver
        )
        return self.solution

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "kind": "upper_bound",
            "parameters": {
                "horizon_hours": self.model.horizon,
                "cyclic_soc": True,
                "initial_soc": "free (chosen by the optimiser)",
            },
            "caveat": (
                "Consumes the realised future over the entire year. A ceiling "
                "on what any controller could achieve, not a design."
            ),
        }


@dataclass
class ScheduleController:
    """Replays a precomputed power schedule through the real dispatcher.

    The bridge between planning and simulation. Running a plan through the
    same dispatcher every other controller uses is what turns "the LP says
    X compute" into "the simulator agrees it is X compute" -- and any gap
    between the two is a modelling error we want to see, not hide.
    """

    schedule_mw: np.ndarray
    name: str = "schedule"

    def reset(self, *, horizon: int) -> None:
        if len(self.schedule_mw) < horizon:
            raise ValueError(
                f"Schedule has {len(self.schedule_mw)} steps, need {horizon}"
            )

    def choose_power(self, obs: Observation) -> float:
        return float(self.schedule_mw[obs.t])

    def metadata(self) -> dict:
        return {"name": self.name, "kind": "replay", "parameters": {"steps": len(self.schedule_mw)}}


@dataclass
class ForecastMPCController:
    """Receding-horizon MPC that plans against whatever it is told to believe.

    At each hour: observe SOC, ask the forecast for the next ``horizon_hours``
    of solar, solve the LP, execute only the first action, advance, re-solve.

    The forecast is the *only* channel through which the future reaches this
    controller. Hand it :class:`~flexcompute.forecast.PerfectSolarForecast` and
    it becomes the perfect-foresight upper bound; hand it
    :class:`~flexcompute.forecast.NoisySolarForecast` and it becomes a
    deployable design whose plans are wrong in the way real plans are wrong.
    Nothing else in the class changes, which is exactly what makes the
    difference between the two runs attributable to *information* and not to
    the controller.

    Two properties make that substitution safe rather than merely convenient:
    the inverter cap is re-derived from the believed series each hour (see
    :meth:`PlantModel.solar_to_bus_max`), and the executed action is the first
    hour of the plan, which the forecast reports without error. Everything the
    controller is wrong about, it is wrong about *in the plan* -- and gets to
    revise before acting on it.

    The terminal value of stored energy stops the controller emptying the
    battery on the last hour of every horizon. It is priced at what that energy
    would yield if spent at the fleet's most compute-efficient operating point,
    scaled by ``terminal_value_scale``. That scale is the only tuning knob in
    the whole controller, and results are reported with a sweep over it.
    """

    model: PlantModel
    fleet: GpuFleet
    forecast: SolarForecast
    horizon_hours: int = 48
    terminal_value_scale: float = 0.95
    solver: str = "HIGHS"
    name: str = "forecast_mpc"

    _solve_count: int = field(default=0, init=False, repr=False)
    _terminal_value: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._terminal_value = (
            self.terminal_value_scale * self.model.marginal_compute_per_battery_mwh()
        )

    def reset(self, *, horizon: int) -> None:
        self._solve_count = 0

    def choose_power(self, obs: Observation) -> float:
        # The forecast is the controller's *belief* about solar; the inverter
        # cap is re-derived from that belief rather than from the truth, so a
        # non-perfect forecast plugs in unchanged.
        window = self.model.window(obs.t, self.horizon_hours)
        window = window.with_solar(self.forecast.horizon(obs.t, window.horizon))
        solution = solve_schedule(
            window,
            initial_soc_mwh=obs.soc_mwh,
            terminal_value_per_mwh=self._terminal_value,
            solver=self.solver,
        )
        self._solve_count += 1
        return float(solution.it_power_mw[0])

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "kind": "mpc",
            "parameters": {
                "horizon_hours": self.horizon_hours,
                "terminal_value_scale": self.terminal_value_scale,
                "terminal_value_compute_per_mwh": self._terminal_value,
                "solver": self.solver,
                "solves": self._solve_count,
            },
            "forecast": self.forecast.metadata(),
        }


@dataclass
class PerfectForesightMPCController(ForecastMPCController):
    """The same controller, restricted to exact knowledge of future solar.

    Identical machinery to :class:`ForecastMPCController` -- it exists to make
    perfect foresight a thing you have to *ask for by name*. The type check is
    the enforcement: a run labelled ``perfect_foresight_mpc`` cannot quietly
    have been given a noisy forecast, and a noisy run cannot quietly inherit
    the perfect-foresight label. Design rule 4, made mechanical.

    Results from this class are an upper bound and must be reported as one.
    """

    name: str = "perfect_foresight_mpc"

    def __post_init__(self) -> None:
        if not isinstance(self.forecast, PerfectSolarForecast):
            raise TypeError(
                "PerfectForesightMPCController requires a PerfectSolarForecast; "
                f"got {type(self.forecast).__name__}. Use ForecastMPCController "
                "for any forecast that can be wrong."
            )
        super().__post_init__()
