"""Capital cost model with battery power and energy priced separately.

Why upstream's cost model cannot be reused as-is
------------------------------------------------
The reference model quotes storage capex per kW only, for an implied 4-hour
system (`ASSUMPTIONS` A6). Its sizing search therefore has no reason to prefer
one duration over another -- an optimiser allowed to vary duration under that
model would buy unlimited MWh for free. Experiment B varies duration, so the
cost model has to price it.

How the split is derived
------------------------
NREL/NLR's utility-scale battery storage cost work states the decomposition
directly:

    Total system cost ($/kW) = Battery pack cost ($/kWh) x Duration (h)
                               + BOS cost ($/kW)

and publishes total installed cost across durations. From the 2/4/6/8/10-hour
row of that table (403 / 574 / 744 / 915 / 1086 $/kW) the affine fit is exact to
within a dollar:

    slope     = (574 - 403) / 2  = 85.5 $/kWh      (energy component)
    intercept = 403 - 2 x 85.5   = 232.0 $/kW      (power component)

    check: 85.5 x 6 + 232 = 745 (vs 744);  x 8 -> 916 (vs 915);  x 10 -> 1087 (vs 1086)

We use only the **ratio** from that source. Absolute levels stay upstream's, so
the 4-hour total is preserved exactly and every Milestone-1 cost figure is
unchanged. `tests/test_costs.py` asserts that reconciliation numerically rather
than trusting this docstring.

Source: NLR/NREL, *Cost Projections for Utility-Scale Battery Storage: 2023
Update*, https://docs.nlr.gov/docs/fy23osti/85332.pdf

Scope
-----
Year-0 capital only. No O&M, no battery replacement, no discounting, no
degradation. That is deliberate for Experiment B, which asks "how much less
plant for the same compute" -- a question about capital, where the omitted terms
scale similarly for every strategy. It is **not** an LCOE and must not be quoted
as one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

#: Energy and power components read off the NLR/NREL duration table (2022 USD).
#: Only their ratio is used; see the module docstring.
NREL_ENERGY_COST_PER_KWH = 85.5
NREL_POWER_COST_PER_KW = 232.0
NREL_REFERENCE_DURATION_H = 4.0

SQ_KM_PER_ACRE = 0.00404686


def energy_share_at_reference_duration() -> float:
    """Fraction of a 4-hour system's capex attributable to stored energy."""
    energy = NREL_ENERGY_COST_PER_KWH * NREL_REFERENCE_DURATION_H
    return energy / (energy + NREL_POWER_COST_PER_KW)


@dataclass(frozen=True)
class CapitalCostModel:
    """Year-0 capex for a (solar MW, battery MW, battery MWh) design.

    Storage is split into a $/kW term and a $/kWh term whose combination at
    ``reference_duration_h`` reproduces upstream's quoted per-kW figure exactly.
    """

    solar_cost_per_kw: float          # modules + solar share of BOS
    storage_cost_per_kw_at_ref: float # upstream's BESS + battery BOS, per kW
    energy_share: float               # of the reference-duration total
    reference_duration_h: float = NREL_REFERENCE_DURATION_H
    land_cost_per_sq_km: float = 0.0
    solar_acres_per_mw: float = 0.0
    battery_acres_per_mwh: float = 0.0

    # -- decomposed storage prices ----------------------------------------
    @property
    def storage_cost_per_kwh(self) -> float:
        total = self.storage_cost_per_kw_at_ref
        return self.energy_share * total / self.reference_duration_h

    @property
    def storage_cost_per_kw(self) -> float:
        return (1.0 - self.energy_share) * self.storage_cost_per_kw_at_ref

    # -- capex -------------------------------------------------------------
    def solar_capex_usd(self, solar_mw: float) -> float:
        return solar_mw * 1000.0 * self.solar_cost_per_kw

    def storage_capex_usd(self, battery_mw: float, battery_mwh: float) -> float:
        return (
            battery_mw * 1000.0 * self.storage_cost_per_kw
            + battery_mwh * 1000.0 * self.storage_cost_per_kwh
        )

    def land_acres(self, solar_mw: float, battery_mwh: float) -> float:
        return solar_mw * self.solar_acres_per_mw + battery_mwh * self.battery_acres_per_mwh

    def land_capex_usd(self, solar_mw: float, battery_mwh: float) -> float:
        return self.land_acres(solar_mw, battery_mwh) * SQ_KM_PER_ACRE * self.land_cost_per_sq_km

    def total_capex_usd(self, solar_mw: float, battery_mw: float, battery_mwh: float) -> float:
        return (
            self.solar_capex_usd(solar_mw)
            + self.storage_capex_usd(battery_mw, battery_mwh)
            + self.land_capex_usd(solar_mw, battery_mwh)
        )

    def breakdown(self, solar_mw: float, battery_mw: float, battery_mwh: float) -> dict:
        return {
            "solar_musd": self.solar_capex_usd(solar_mw) / 1e6,
            "storage_musd": self.storage_capex_usd(battery_mw, battery_mwh) / 1e6,
            "land_musd": self.land_capex_usd(solar_mw, battery_mwh) / 1e6,
            "total_musd": self.total_capex_usd(solar_mw, battery_mw, battery_mwh) / 1e6,
            "land_acres": self.land_acres(solar_mw, battery_mwh),
        }

    def metadata(self) -> dict:
        return {
            **asdict(self),
            "storage_cost_per_kw_derived": self.storage_cost_per_kw,
            "storage_cost_per_kwh_derived": self.storage_cost_per_kwh,
            "basis": "year-0 capital only; no O&M, replacement, discounting or degradation",
            "source_ratio": "NLR/NREL Cost Projections for Utility-Scale Battery Storage (2023 update)",
        }


def from_upstream_config(config, architecture: str = "ac_coupled") -> CapitalCostModel:
    """Build the cost model from the reference project's own parameters.

    Every absolute price comes from upstream. The only thing added is *how* the
    storage price divides between power and energy.
    """
    costs = config.costs
    is_dc = architecture == "dc_coupled"
    return CapitalCostModel(
        solar_cost_per_kw=costs.solar_cost_y0
        + (costs.solar_bos_cost_y0_dc if is_dc else costs.solar_bos_cost_y0_ac),
        storage_cost_per_kw_at_ref=costs.bess_cost_y0
        + (costs.battery_bos_cost_y0_dc if is_dc else costs.battery_bos_cost_y0_ac),
        energy_share=energy_share_at_reference_duration(),
        land_cost_per_sq_km=costs.land_cost_per_sq_km,
        solar_acres_per_mw=config.design.solar_acres_per_mw,
        battery_acres_per_mwh=config.design.battery_acres_per_mwh,
    )


def upstream_equivalent_capex_usd(
    config, solar_mw: float, battery_mw: float, architecture: str = "ac_coupled"
) -> float:
    """Upstream's own capex formula, for reconciliation.

    Upstream prices storage per kW for an implied 4-hour system and adds land.
    Our decomposed model must return the same number at 4 hours.
    """
    costs = config.costs
    is_dc = architecture == "dc_coupled"
    solar = solar_mw * 1000.0 * (
        costs.solar_cost_y0 + (costs.solar_bos_cost_y0_dc if is_dc else costs.solar_bos_cost_y0_ac)
    )
    storage = battery_mw * 1000.0 * (
        costs.bess_cost_y0 + (costs.battery_bos_cost_y0_dc if is_dc else costs.battery_bos_cost_y0_ac)
    )
    acres = (
        solar_mw * config.design.solar_acres_per_mw
        + battery_mw * NREL_REFERENCE_DURATION_H * config.design.battery_acres_per_mwh
    )
    land = acres * SQ_KM_PER_ACRE * costs.land_cost_per_sq_km
    return solar + storage + land
