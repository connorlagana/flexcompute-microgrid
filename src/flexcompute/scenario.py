"""Scenario definition and site construction.

A :class:`Scenario` is the complete, hashable description of an experiment's
*controller-independent* inputs: where the site is, how big the GPU fleet is,
which electrical architecture is used, which weather source is trusted, and
which random seed governs the optimiser.

:meth:`Scenario.build` turns that into a :class:`Site` -- weather, cooling
selection, hourly PUE, the nameplate IT load profile, and the normalised solar
production profile. Everything in a ``Site`` is fixed physics and hardware.
Nothing in it depends on how the GPUs are operated.

That separation is the apples-to-apples guarantee for the eventual controller
comparison: every strategy is handed the *same* ``Site`` object, so any
difference in outcome is attributable to the control policy alone.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .upstream_bridge import (
    UPSTREAM_TABLES,
    ensure_upstream_importable,
    load_upstream_config,
    upstream_commit,
    upstream_workdir,
)
from .weather import TMY, get_tmy

logger = logging.getLogger(__name__)

HOURS_PER_YEAR = 8760

#: Location presets, matching upstream's ``analysis_wrapper.LOCATIONS``.
LOCATIONS: dict[str, tuple[float, float]] = {
    "el_paso": (31.77, -106.46),
    "phoenix": (33.45, -112.07),
    "dallas": (32.78, -96.80),
    "seattle": (47.61, -122.33),
    "chicago": (41.88, -87.63),
}


@dataclass(frozen=True)
class Scenario:
    """Everything needed to reproduce a site, exactly."""

    location: str = "dallas"
    total_gpus: int = 10_000
    required_uptime_pct: float = 99.0
    architecture: str = "ac_coupled"   # ac_coupled | dc_coupled
    topology: str = "mv_coupled"       # mv_coupled | lv_direct
    weather_source: str = "pvgis"      # pvgis | nsrdb
    seed: int = 20260815

    @property
    def latitude(self) -> float:
        return LOCATIONS[self.location][0]

    @property
    def longitude(self) -> float:
        return LOCATIONS[self.location][1]

    def key(self) -> str:
        """Short stable digest, used to name snapshot files."""
        blob = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def label(self) -> str:
        return (
            f"{self.location}_{self.total_gpus // 1000}kgpu_"
            f"{self.architecture}_{self.topology}_{self.weather_source}"
        )

    def build(self, *, refresh_weather: bool = False) -> "Site":
        return build_site(self, refresh_weather=refresh_weather)


@dataclass
class Site:
    """Controller-independent physical model of one site-year.

    Attributes
    ----------
    facility_load
        Upstream ``FacilityLoad``. Its ``hourly_it_load_mw`` is the *nameplate*
        (fixed-load) IT demand -- the profile the reference model treats as
        exogenous, and the profile a flexible controller is allowed to depart
        from.
    solar_profile
        Upstream-format frame with a ``p_dc`` column normalised to the annual
        maximum DC output, so ``p_dc * solar_capacity_mw`` gives hourly DC MW.
    hourly_pue
        Weather-driven PUE, 8760 values. Note this is a function of ambient
        conditions only -- it does not respond to IT load. See ASSUMPTIONS.md.
    """

    scenario: Scenario
    config: object                      # upstream Config
    tmy: TMY
    facility_load: object               # upstream FacilityLoad
    solar_profile: pd.DataFrame
    hourly_pue: np.ndarray
    cooling_case: int
    annual_pue: float

    # -- convenience views -------------------------------------------------
    @property
    def it_load_mw(self) -> np.ndarray:
        """Nameplate hourly IT load (MW), 8760 values."""
        return np.asarray(self.facility_load.hourly_it_load_mw, dtype=float)

    @property
    def it_load_avg_mw(self) -> float:
        return float(self.facility_load.it_load_avg_mw)

    @property
    def it_load_max_mw(self) -> float:
        return float(self.facility_load.it_load_max_mw)

    @property
    def solar_p_dc(self) -> np.ndarray:
        """Normalised solar DC output, 0..1 of annual peak, 8760 values."""
        return np.asarray(self.solar_profile["p_dc"].to_numpy(), dtype=float)

    def provenance(self) -> dict:
        return {
            "scenario": asdict(self.scenario),
            "scenario_key": self.scenario.key(),
            "upstream_commit": upstream_commit(),
            "weather": self.tmy.provenance(),
            "weather_summary": self.tmy.summary(),
            "cooling_case": int(self.cooling_case),
            "annual_pue": float(self.annual_pue),
            "solar_capacity_factor_of_peak": float(self.solar_p_dc.mean()),
            "it_load_avg_mw": self.it_load_avg_mw,
            "it_load_max_mw": self.it_load_max_mw,
            "facility_load_design_mw": float(self.facility_load.facility_load_design_mw),
        }


def build_site(scenario: Scenario, *, refresh_weather: bool = False) -> Site:
    """Assemble a :class:`Site` from a :class:`Scenario`.

    Weather is injected rather than fetched by upstream. Upstream's
    ``DatacenterAnalyzer`` fetches TMY only when ``self.weather_df is None``,
    and hands that same frame to ``FacilityLoad.tmy_weather``, which
    ``get_solar_generation`` then reuses. Setting ``weather_df`` up front
    therefore drives cooling selection, PUE, design temperature, and PV
    production from one frame with zero network access and zero upstream edits.
    """
    ensure_upstream_importable()
    from datacenter_analyzer import DatacenterAnalyzer  # type: ignore[import-not-found]
    from pvstoragesim import get_solar_generation       # type: ignore[import-not-found]

    tmy = get_tmy(
        scenario.latitude,
        scenario.longitude,
        scenario.weather_source,
        refresh=refresh_weather,
    )
    cfg = load_upstream_config()

    analyzer = DatacenterAnalyzer(
        latitude=scenario.latitude,
        longitude=scenario.longitude,
        total_gpus=scenario.total_gpus,
        lookup_dir=str(UPSTREAM_TABLES),
        config=cfg,
    )
    analyzer.weather_df = tmy.data           # <- injection point
    analyzer.analyze_cooling()

    # CWD only matters here: this path reads the hourly IT load CSV through a
    # root-relative default that DatacenterAnalyzer does not expose.
    with upstream_workdir():
        facility_load = analyzer.calculate_facility_load(
            required_uptime_pct=scenario.required_uptime_pct
        )

    solar_profile = get_solar_generation(
        scenario.latitude, scenario.longitude, facility_load=facility_load
    )

    _validate_site_arrays(facility_load, solar_profile, analyzer.hourly_pue)

    return Site(
        scenario=scenario,
        config=cfg,
        tmy=tmy,
        facility_load=facility_load,
        solar_profile=solar_profile,
        hourly_pue=np.asarray(analyzer.hourly_pue, dtype=float),
        cooling_case=analyzer.pue_results["optimal_case"],
        annual_pue=float(analyzer.annual_pue),
    )


def _validate_site_arrays(facility_load, solar_profile, hourly_pue) -> None:
    for name, arr in (
        ("hourly_it_load_mw", facility_load.hourly_it_load_mw),
        ("hourly_pue", hourly_pue),
        ("solar p_dc", solar_profile["p_dc"].to_numpy()),
    ):
        a = np.asarray(arr, dtype=float)
        if a.shape != (HOURS_PER_YEAR,):
            raise ValueError(f"{name} must have {HOURS_PER_YEAR} values, got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(np.asarray(hourly_pue) < 1.0):
        raise ValueError("hourly_pue below 1.0 would imply cooling generates power")
