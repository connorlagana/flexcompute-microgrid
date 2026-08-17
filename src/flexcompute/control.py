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
from typing import Protocol, runtime_checkable

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
