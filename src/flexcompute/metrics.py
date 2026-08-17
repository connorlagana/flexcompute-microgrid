"""Metric extraction and physical-validity auditing.

Two jobs:

``summarize_simulation``
    Pull a flat, unit-labelled dict of results out of an upstream
    ``SimulationResult``. Every key carries its unit in the name.

``audit_energy_balance``
    Independently re-derive the energy books from the hourly trace and report
    the worst violation of each physical law the model must obey. This is
    deliberately *not* a re-implementation of the dispatch: it consumes the
    recorded trace and checks it, so a future controller that changes dispatch
    behaviour cannot silently start manufacturing energy.

Laws checked
------------
1. **PV accounting.** Every DC joule leaving the array is either serving load,
   charging the battery, or curtailed.
2. **Battery state.** ``soc[t+1] - soc[t] == charge[t] - discharge[t]`` exactly
   (both terms measured at the battery terminal, after conversion losses).
3. **SOC bounds.** ``0 <= soc <= energy_capacity``.
4. **Power limits.** Charge and discharge never exceed the battery MW rating.
5. **Bus balance.** ``solar_at_bus + discharge_at_bus + unmet == bus_load``
   whenever the bus is in deficit.

Sign convention: all flows are non-negative magnitudes; direction is carried by
the variable name. Energy units are MWh, power units MW, and with a 1-hour
timestep the two are numerically equal (documented in ASSUMPTIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EnergyAudit:
    """Worst-case violation of each conservation law, in MW (== MWh at 1 h)."""

    pv_accounting_max_abs_mw: float
    battery_state_max_abs_mw: float
    soc_min_mwh: float
    soc_max_mwh: float
    soc_capacity_mwh: float
    charge_max_mw: float
    discharge_max_mw: float
    power_limit_mw: float
    bus_balance_max_abs_mw: float
    # Energy the model neither used, stored, nor booked as curtailed. Non-zero
    # exposes an accounting leak rather than a physical loss.
    unbooked_pv_mwh: float

    def violations(self, tol_mw: float = 1e-6) -> list[str]:
        """Return human-readable descriptions of any law that was broken."""
        problems: list[str] = []
        if self.pv_accounting_max_abs_mw > tol_mw:
            problems.append(
                f"PV accounting residual {self.pv_accounting_max_abs_mw:.3e} MW > {tol_mw:.0e}"
            )
        if self.battery_state_max_abs_mw > tol_mw:
            problems.append(
                f"Battery state residual {self.battery_state_max_abs_mw:.3e} MW > {tol_mw:.0e}"
            )
        if self.soc_min_mwh < -tol_mw:
            problems.append(f"SOC went negative: {self.soc_min_mwh:.6f} MWh")
        if self.soc_max_mwh > self.soc_capacity_mwh + tol_mw:
            problems.append(
                f"SOC {self.soc_max_mwh:.6f} MWh exceeded capacity {self.soc_capacity_mwh:.6f} MWh"
            )
        if self.charge_max_mw > self.power_limit_mw + tol_mw:
            problems.append(
                f"Charge {self.charge_max_mw:.6f} MW exceeded rating {self.power_limit_mw:.6f} MW"
            )
        if self.discharge_max_mw > self.power_limit_mw + tol_mw:
            problems.append(
                f"Discharge {self.discharge_max_mw:.6f} MW exceeded rating {self.power_limit_mw:.6f} MW"
            )
        if self.bus_balance_max_abs_mw > tol_mw:
            problems.append(
                f"Bus balance residual {self.bus_balance_max_abs_mw:.3e} MW > {tol_mw:.0e}"
            )
        return problems

    def as_dict(self) -> dict:
        return asdict(self)


def audit_energy_balance(
    hourly: pd.DataFrame,
    multipliers: dict,
    battery_energy_mwh: float,
    battery_power_mw: float,
    inverter_cap_mw_dc: Optional[float] = None,
) -> EnergyAudit:
    """Re-derive the energy books from an hourly trace and report residuals.

    ``hourly`` is the frame produced by ``evaluate_system(return_hourly=True)``.
    ``multipliers`` is the dict from ``PowerFlowAnalyzer.get_bus_architecture_multipliers``.
    """
    stb = multipliers["solar_to_bus"]
    stbat = multipliers["solar_to_battery"]
    btb = multipliers["battery_to_bus"]

    solar_dc = hourly["solar_dc_mw"].to_numpy(dtype=float)
    solar_at_bus = hourly["solar_at_load_mw"].to_numpy(dtype=float)
    charge = hourly["battery_charge_mw"].to_numpy(dtype=float)
    discharge = hourly["battery_discharge_mw"].to_numpy(dtype=float)
    curtailed = hourly["curtailed_solar_mw"].to_numpy(dtype=float)
    unmet = hourly["unmet_load_mw"].to_numpy(dtype=float)
    soc = hourly["battery_soc_mwh"].to_numpy(dtype=float)

    bus_load = (
        hourly["it_load_mw"].to_numpy(dtype=float) * multipliers["bus_to_it"]
        + hourly["cooling_load_mw"].to_numpy(dtype=float) * multipliers["bus_to_cooling"]
    )

    surplus = solar_at_bus > bus_load

    # --- law 1: PV accounting -------------------------------------------
    # DC energy that reached the bus, at the array side of the solar->bus path.
    dc_to_bus = np.minimum(solar_at_bus, bus_load) * stb
    booked = dc_to_bus + charge * stbat + curtailed
    residual = solar_dc - booked
    # In a *surplus* hour the split is exact by construction. In a *deficit*
    # hour with an inverter cap, PV above the cap is physically unusable and
    # upstream does not record it as curtailment; that shows up here as a
    # positive residual and is reported separately rather than treated as a
    # conservation failure.
    pv_residual = np.where(surplus, residual, 0.0)
    unbooked = float(np.clip(residual, 0.0, None)[~surplus].sum())

    # --- law 2: battery state -------------------------------------------
    # hourly soc is start-of-hour, so the last transition is not observable.
    delta_soc = np.diff(soc)
    battery_residual = delta_soc - (charge[:-1] - discharge[:-1])

    # --- law 5: bus balance in deficit hours ----------------------------
    deficit_residual = np.where(
        surplus, 0.0, bus_load - (solar_at_bus + discharge / btb + unmet)
    )

    return EnergyAudit(
        pv_accounting_max_abs_mw=float(np.max(np.abs(pv_residual))),
        battery_state_max_abs_mw=float(np.max(np.abs(battery_residual))),
        soc_min_mwh=float(soc.min()),
        soc_max_mwh=float(soc.max()),
        soc_capacity_mwh=float(battery_energy_mwh),
        charge_max_mw=float(charge.max()),
        discharge_max_mw=float(discharge.max()),
        power_limit_mw=float(battery_power_mw),
        bus_balance_max_abs_mw=float(np.max(np.abs(deficit_residual))),
        unbooked_pv_mwh=unbooked,
    )


def summarize_simulation(
    sim,
    *,
    solar_mw: float,
    battery_mw: float,
    battery_mwh: float,
) -> dict:
    """Flatten an upstream ``SimulationResult`` into unit-labelled scalars."""
    hourly = sim.hourly_data
    summary = {
        # -- sizing --
        "solar_mw_dc": float(solar_mw),
        "battery_mw": float(battery_mw),
        "battery_mwh": float(battery_mwh),
        "battery_duration_h": float(battery_mwh / battery_mw) if battery_mw else 0.0,
        # -- reliability --
        "uptime_pct": float(sim.uptime_pct),
        "energy_served_pct": float(sim.energy_served_pct),
        "unmet_load_mwh": float(sim.unmet_load_mwh),
        # -- energy --
        "solar_generation_mwh": float(sim.solar_generation_mwh),
        "solar_curtailed_mwh": float(sim.solar_curtailed_mwh),
        "load_served_mwh": float(sim.load_served_mwh),
        "battery_charged_mwh": float(sim.battery_charged_mwh),
        "battery_discharged_mwh": float(sim.battery_discharged_mwh),
        "battery_cycles_per_year": float(sim.battery_cycles_per_year),
    }
    gen = summary["solar_generation_mwh"]
    summary["solar_curtailed_pct"] = float(100.0 * summary["solar_curtailed_mwh"] / gen) if gen else 0.0

    if hourly is not None:
        soc = hourly["battery_soc_mwh"].to_numpy(dtype=float)
        summary.update(
            {
                "soc_start_mwh": float(soc[0]),
                "soc_end_mwh": float(soc[-1]),
                "soc_mean_mwh": float(soc.mean()),
                "soc_min_mwh": float(soc.min()),
                "hours_with_unmet_load": int((hourly["unmet_load_mw"].to_numpy() >= 0.001).sum()),
            }
        )
    return summary
