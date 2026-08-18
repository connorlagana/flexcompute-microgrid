"""Compute controllers: the decision layer.

A controller answers one question, once per timestep: *how much power should
the GPU fleet draw right now?*

The four power quantities
-------------------------
Keeping these distinct is what lets us tell a deliberate slowdown apart from a
blackout. They are never collapsed into a single "load" number.

``nameplate``
    What a fixed-load facility of this size would draw. The reference model's
    exogenous profile. A *design* concept — it is what the infrastructure was
    sized against.
``unconstrained_demand``
    What the workload actually wants right now, absent any energy limit. A
    *workload* concept. Currently equal to nameplate, but separate by design:
    later work (deadlines, oversubscription, deferred queues) will make them
    diverge, and the accounting must already be ready for that.
``controller_target``
    What the controller asks for. The only thing a controller influences.
``delivered``
    What the bus actually supplied. Physics decides this, not the controller.

Two gaps follow, and they mean opposite things:

    voluntary_throttle   = unconstrained_demand - controller_target
    involuntary_shortfall = controller_target   - delivered

The first is a *choice*: energy deliberately not consumed, banked for later.
The second is a *failure*: the controller asked for power the system could not
deliver. A good controller trades the first for capital savings while keeping
the second near zero. A controller that scores well only by racking up the
second is broken, and this decomposition is what exposes it.

Causality
---------
:class:`Observation` carries only what is knowable at time ``t``: current state,
current measurements, and the controller's own history. There is no field
holding future solar. A controller that wants to anticipate must be handed a
``SolarForecast`` object explicitly (Milestone 4), which makes perfect foresight
an opt-in dependency rather than an accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Observation:
    """Everything a controller may legitimately know at timestep ``t``."""

    t: int                          # hour of year, 0..8759, local standard time
    hour_of_day: int                # 0..23, local standard time
    soc_mwh: float                  # battery stored energy now
    soc_fraction: float             # soc_mwh / battery_energy_mwh, 0..1
    battery_energy_mwh: float       # nameplate energy capacity
    battery_power_mw: float         # nameplate charge/discharge rating
    solar_dc_mw: float              # PV available *this hour* (measured, not forecast)
    pue: float                      # this hour's PUE
    nameplate_it_mw: float          # what a fixed-load facility would draw
    unconstrained_demand_mw: float  # what the workload wants
    min_operating_it_mw: float      # below this the fleet parks (curve domain floor)

    @property
    def max_it_mw(self) -> float:
        """No point requesting more power than there is work to do."""
        return self.unconstrained_demand_mw


@runtime_checkable
class ComputeController(Protocol):
    """Chooses GPU power. Implementations must be deterministic."""

    name: str

    def reset(self, *, horizon: int) -> None:
        """Clear per-run state. Called once before each simulation."""
        ...

    def choose_power(self, obs: Observation) -> float:
        """Return the requested IT power in MW for this timestep."""
        ...

    def metadata(self) -> dict:
        """Serialisable description, recorded with every result."""
        ...


@dataclass
class FixedLoadController:
    """Reproduces the reference model exactly: always ask for full demand.

    The gate for Milestone 2 is that this controller yields
    ``controller_target == unconstrained_demand`` at every timestep, and that
    routing it through the closed-loop simulator reproduces upstream's dispatch
    bit-for-bit.
    """

    name: str = "fixed_load"

    def reset(self, *, horizon: int) -> None:
        return None

    def choose_power(self, obs: Observation) -> float:
        return obs.unconstrained_demand_mw

    def metadata(self) -> dict:
        return {"name": self.name, "kind": "baseline", "parameters": {}}


@dataclass
class SimpleThrottleController:
    """Deliberately naive SOC-banded throttling.

    Not optimal and not intended to be. Its only job is to demonstrate that
    demand can react to stored energy, and to give the MPC something honest to
    beat. It is myopic in the strongest sense: it looks at the battery and
    nothing else — not the time of day, not the current sunshine, not the
    season, not tomorrow.

    Bands are ``(soc_fraction_threshold, power_fraction)`` pairs evaluated
    highest-threshold-first. Below the lowest threshold the fleet parks.

    The defaults are chosen to be obviously unsophisticated: full power when
    the battery is more than half full, then coarse steps down.
    """

    bands: tuple[tuple[float, float], ...] = (
        (0.50, 1.00),   # over half full: run flat out
        (0.30, 0.75),   # getting low: back off
        (0.15, 0.50),   # low: back off hard
    )
    park_below: float = 0.15
    name: str = "simple_throttle"

    def __post_init__(self) -> None:
        thresholds = [b[0] for b in self.bands]
        if thresholds != sorted(thresholds, reverse=True):
            raise ValueError("bands must be ordered by descending SOC threshold")
        if any(not (0.0 <= p <= 1.0) for _, p in self.bands):
            raise ValueError("band power fractions must lie in [0, 1]")

    def reset(self, *, horizon: int) -> None:
        return None

    def choose_power(self, obs: Observation) -> float:
        if obs.soc_fraction < self.park_below:
            return 0.0          # dispatcher snaps this to idle
        for threshold, power_fraction in self.bands:
            if obs.soc_fraction >= threshold:
                return power_fraction * obs.unconstrained_demand_mw
        return 0.0

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "kind": "heuristic",
            "parameters": {
                "bands": [list(b) for b in self.bands],
                "park_below": self.park_below,
            },
        }


# ---------------------------------------------------------------------------
# Plant constants a controller may know without knowing the future
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlantConstants:
    """Nameplate electrical facts about the plant. Scalars only, by design.

    A controller that wants to convert "MW available at the bus" into "MW the
    GPUs may draw" needs the conversion efficiencies and the inverter rating.
    Those are datasheet numbers — an operator knows them at commissioning — so
    handing them to a controller grants no foresight.

    The **scalars only** rule is the enforcement. There is deliberately no
    field here holding an hourly series: no solar profile, no PUE profile, no
    demand profile. A controller given a :class:`PlantConstants` therefore
    *cannot* read ahead even by accident, because there is nothing to read. Any
    time-varying quantity it needs must arrive through :class:`Observation`,
    which only ever describes the current hour.

    Contrast :class:`~flexcompute.mpc.PlantModel`, which does carry annual
    series and is consequently only safe to hand to a controller that also has
    an explicit forecast object.
    """

    m_it: float                       # bus MW per IT MW
    m_cool: float                     # bus MW per cooling MW
    m_solar_bus: float                # PV DC MW per MW delivered at the bus
    m_batt_bus: float                 # battery MW drawn per MW delivered at the bus
    cooling_fixed_fraction: float = 0.0
    inverter_cap_mw_dc: Optional[float] = None

    def solar_at_bus_mw(self, solar_dc_mw: float) -> float:
        """PV power reaching the bus this hour, after inverter clipping."""
        reachable = (
            solar_dc_mw if self.inverter_cap_mw_dc is None
            else min(solar_dc_mw, self.inverter_cap_mw_dc)
        )
        return reachable / self.m_solar_bus

    def bus_load_coefficients(self, obs: "Observation") -> tuple[float, float]:
        """``(alpha, beta)`` such that bus load == ``alpha * it_mw + beta``.

        Transcribed from the dispatcher's cooling model so the governor plans
        against the same physics it will be judged by. ``beta`` is the cooling
        floor that is drawn regardless of GPU power; it vanishes when cooling is
        purely proportional (``cooling_fixed_fraction == 0``).
        """
        nameplate_cooling_mw = obs.nameplate_it_mw * (obs.pue - 1.0)
        beta = self.cooling_fixed_fraction * nameplate_cooling_mw * self.m_cool
        per_it = (
            (1.0 - self.cooling_fixed_fraction) * nameplate_cooling_mw
            / obs.unconstrained_demand_mw
            if obs.unconstrained_demand_mw > 0.0 else 0.0
        )
        return self.m_it + per_it * self.m_cool, beta

    def as_dict(self) -> dict:
        return {
            "m_it": self.m_it,
            "m_cool": self.m_cool,
            "m_solar_bus": self.m_solar_bus,
            "m_batt_bus": self.m_batt_bus,
            "cooling_fixed_fraction": self.cooling_fixed_fraction,
            "inverter_cap_mw_dc": self.inverter_cap_mw_dc,
        }


# ---------------------------------------------------------------------------
# The Handmer-style governor
# ---------------------------------------------------------------------------

@dataclass
class CaseyGovernor:
    """A minimum-viable governor in the style described by Casey Handmer.

    Source: Casey Handmer, *Direct Current Data Centers*, 30 January 2026,
    https://caseyhandmer.wordpress.com/2026/01/30/direct-current-data-centers/

    What the source actually says
    -----------------------------
    Four statements, quoted, are all the specification there is:

    1. "Throw in a basic 'governor' that throttles the GPU when it predicts the
       battery will be exhausted before dawn."
    2. "It merely assesses the time of day, the state of the battery and of
       solar generation and curtails GPU utilization accordingly."
    3. "This governor is not very sophisticated, for example, it has no ability
       to take weather prediction into account."
    4. "Note how the governor throttles output early on the fourth day by
       rationing power until the following morning. The cubic power consumption
       of GPUs means that throttling a little bit early is much better for token
       production than running full blast into a wall and then dropping to zero
       production until the sun comes back up."

    The implementation below is the simplest rule satisfying all four:

        allowance = usable stored energy / hours until the next sunrise
        target    = whatever GPU power (solar now + allowance) can support

    Statement 1 gives the objective (do not be empty before dawn). Statement 2
    gives the input set, and it is exactly the input set used here — clock,
    battery, present generation, and nothing else. Statement 3 is enforced
    structurally: this class has no ``SolarForecast`` field, so it cannot be
    handed one. Statement 4 gives the behaviour, including the detail that
    throttling begins *during* a bad day rather than at nightfall, which is why
    :func:`~flexcompute.solar_clock.hours_until_next_sunrise` points at
    tomorrow's sunrise while the sun is up.

    What we had to approximate
    --------------------------
    The source publishes no formula, so these choices are ours and are recorded
    in ASSUMPTIONS B14:

    * **Constant-rate rationing.** Spending the battery evenly over the hours to
      sunrise is the simplest reading of "rationing power until the following
      morning". The source neither states nor rules out a rate that varies
      through the night.
    * **"Solar has returned" is a threshold on present output.** Rationing stops
      when PV alone can carry ``solar_return_fraction`` of the full-power bus
      load. The source says the governor assesses "the state of solar
      generation" but not how.
    * **The horizon is the ephemeris sunrise**, from solar geometry alone. This
      is a calendar, not a forecast — see :mod:`flexcompute.solar_clock`.
    * **Zero reserve by default.** The governor plans to arrive at dawn empty,
      which is what "exhausted before dawn" implies as the thing to avoid. A
      reserve knob exists for sensitivity but is off.
    * **The GPU curve enters only as a floor.** The source's "cubic power
      consumption" is the *reason* rationing beats running flat out, and this
      model carries that concavity in the fleet curve rather than in the
      governor. The governor consults the curve for one decision only: whether
      the sustainable power is so low that the fleet should park instead of
      being asked to operate below its measured domain.

    What this is not
    ----------------
    Not an optimiser. It does not price stored energy, does not look beyond the
    next sunrise, and will happily ration against a night that turns out to be
    followed by a week of overcast. Beating it is the MPC's job; the gap between
    the two is the value of a weather forecast, which is the number this whole
    comparison exists to produce.
    """

    plant: PlantConstants
    hours_to_sunrise: np.ndarray
    #: Rationing stops once PV alone covers this multiple of the full-power bus
    #: load. 1.0 means "the plant can run on sunlight" — the natural reading of
    #: "solar has returned", and the value used for the headline result.
    solar_return_fraction: float = 1.0
    #: Stored energy held back from the nightly ration, as a fraction of
    #: capacity. Zero is the faithful default; see the docstring.
    reserve_fraction: float = 0.0
    name: str = "casey_governor"

    def __post_init__(self) -> None:
        self.hours_to_sunrise = np.asarray(self.hours_to_sunrise, dtype=float)
        if self.hours_to_sunrise.ndim != 1:
            raise ValueError("hours_to_sunrise must be one-dimensional")
        if np.any(self.hours_to_sunrise <= 0):
            raise ValueError("hours_to_sunrise must be strictly positive everywhere")
        if not 0.0 <= self.reserve_fraction < 1.0:
            raise ValueError("reserve_fraction must lie in [0, 1)")
        if self.solar_return_fraction < 0.0:
            raise ValueError("solar_return_fraction must be non-negative")

    def reset(self, *, horizon: int) -> None:
        if len(self.hours_to_sunrise) < horizon:
            raise ValueError(
                f"solar clock covers {len(self.hours_to_sunrise)} hours, need {horizon}"
            )

    def choose_power(self, obs: Observation) -> float:
        demand = obs.unconstrained_demand_mw
        if demand <= 0.0:
            return 0.0

        alpha, beta = self.plant.bus_load_coefficients(obs)
        solar_at_bus = self.plant.solar_at_bus_mw(obs.solar_dc_mw)
        full_bus_load = alpha * demand + beta

        # -- has the sun come back? ---------------------------------------
        # Measured, present-tense, and the only weather input the governor has.
        sun_is_back = solar_at_bus >= self.solar_return_fraction * full_bus_load

        # -- how fast may the battery be spent? ----------------------------
        if sun_is_back:
            # Generation is carrying the plant; nothing to ration against.
            allowance_mw = obs.battery_power_mw
        else:
            usable_mwh = max(
                0.0, obs.soc_mwh - self.reserve_fraction * obs.battery_energy_mwh
            )
            hours = float(self.hours_to_sunrise[obs.t])
            allowance_mw = min(obs.battery_power_mw, usable_mwh / hours)

        # -- what GPU power does that support? -----------------------------
        bus_budget = solar_at_bus + allowance_mw / self.plant.m_batt_bus
        target = (bus_budget - beta) / alpha if alpha > 0.0 else 0.0
        target = min(max(target, 0.0), demand)

        # The one place the GPU curve enters: below the fleet's operating floor
        # there is no measured performance data, so park rather than pretend.
        if target < obs.min_operating_it_mw:
            return 0.0
        return target

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "kind": "heuristic",
            "parameters": {
                "solar_return_fraction": self.solar_return_fraction,
                "reserve_fraction": self.reserve_fraction,
                "plant": self.plant.as_dict(),
                "rationing_horizon": "hours to next ephemeris sunrise",
            },
            "source": {
                "author": "Casey Handmer",
                "title": "Direct Current Data Centers",
                "date": "2026-01-30",
                "url": (
                    "https://caseyhandmer.wordpress.com/2026/01/30/"
                    "direct-current-data-centers/"
                ),
            },
            "caveat": (
                "Reimplementation of a governor described in prose, not "
                "published as code or equations. The constant-rate rationing "
                "rule, the solar-return threshold and the zero reserve are our "
                "approximations; see ASSUMPTIONS B14. Uses no weather forecast."
            ),
        }

