"""Putting every weather year on one physical solar scale.

The problem
-----------
Upstream's ``get_solar_generation`` returns a PV profile normalised by **that
year's own maximum hour**::

    max_dc = np.max(dc_power_values)
    p_dc_normalized = dc_power_values / max_dc

For a single TMY that is harmless bookkeeping. Across many years it is not,
because it means "200 MW-DC of solar" describes a *different physical array* in
every year. A year whose sunniest hour happened to be 5% better than another's
has its entire 8760-hour profile scaled down by 5% relative to that other year,
on the strength of one hour.

Measured at Dallas over 2010-2024 (ERA5), the effect is not small and not
random:

* the divisor ranges from 0.914 to 0.994, so annual output is inflated by
  between +0.6% and +9.4% depending on the year;
* the inflation is largest for the *cloudiest* years, because a cloudy year
  never reaches nameplate and therefore gets divided by the smallest number;
* it re-orders the years. Under per-year normalisation 2012 looks like the
  sunniest year in the record (capacity factor 0.275); on a common scale it is
  mid-pack (0.251), and 2011 is the true leader.

A study whose central output is "how does controller value vary across weather
years" cannot run on a scale that reorders the years.

Why the divisor is nearly always below one
------------------------------------------
pvlib's PVWatts inverter clips AC output at ``pdc0``, which upstream sets to 1.
The PVGIS TMY exceeds nameplate in 79 hours of the year -- cold, clear,
high-irradiance conditions -- so its maximum saturates at exactly 1.0 and the
division does nothing. ERA5 reanalysis years never reach nameplate, because a
reanalysis smooths away precisely those extreme clear-sky hours, so every one of
them gets divided by something less than one.

The fix
-------
Undo the division. The underlying values are already per unit of nameplate DC
capacity, which is the physically meaningful scale and the one on which
"200 MW-DC" means the same array in every year.

:func:`upstream_normalisation_divisor` recovers the factor upstream divided out
by replicating its model chain exactly. ``tests/test_pv_model.py`` pins that
replication bit-for-bit against upstream, so if the reference model's PV
configuration ever changes, this module fails loudly instead of silently
rescaling every result.

For the PVGIS TMY the divisor is exactly ``1.0``, so applying this correction
leaves the committed baseline bit-identical. That is not luck -- it is the
clipping above -- but it is asserted by test rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Transcribed from ``pvstoragesim.get_solar_generation``. Any divergence here
#: silently rescales every result, so it is pinned by test rather than trusted.
UPSTREAM_MODULE_PARAMETERS = {"pdc0": 1, "gamma_pdc": -0.004}
UPSTREAM_INVERTER_PARAMETERS = {"pdc0": 1, "eta_inv_nom": 1}
UPSTREAM_TEMPERATURE_PARAMETERS = {"a": -3.56, "b": -0.075, "deltaT": 3}
UPSTREAM_SYSTEM_TYPE = "single-axis"
UPSTREAM_AOI_MODEL = "physical"
UPSTREAM_SPECTRAL_MODEL = "no_loss"


def unnormalised_dc_profile(
    weather: pd.DataFrame, latitude: float, longitude: float
) -> np.ndarray:
    """PV output per unit of nameplate DC capacity, before any renormalisation.

    A faithful replication of upstream's model chain up to the point where it
    divides by the annual maximum. The returned series is what upstream then
    normalises; multiplying upstream's ``p_dc`` by ``max()`` of this recovers it
    exactly.
    """
    from pvlib import location, modelchain, pvsystem

    array = pvsystem.Array(
        mount=pvsystem.SingleAxisTrackerMount(),
        module_parameters=dict(UPSTREAM_MODULE_PARAMETERS),
        temperature_model_parameters=dict(UPSTREAM_TEMPERATURE_PARAMETERS),
    )
    system = pvsystem.PVSystem(
        arrays=[array], inverter_parameters=dict(UPSTREAM_INVERTER_PARAMETERS)
    )
    chain = modelchain.ModelChain(
        system,
        location.Location(latitude, longitude),
        aoi_model=UPSTREAM_AOI_MODEL,
        spectral_model=UPSTREAM_SPECTRAL_MODEL,
    )
    chain.run_model(weather)
    return np.asarray(chain.results.ac, dtype=float)


def upstream_normalisation_divisor(
    weather: pd.DataFrame, latitude: float, longitude: float
) -> float:
    """The per-year maximum that upstream divides its profile by.

    Multiply upstream's ``p_dc`` by this to get back onto a common,
    per-unit-of-nameplate scale shared by every year and every weather source.
    """
    profile = unnormalised_dc_profile(weather, latitude, longitude)
    peak = float(profile.max())
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"PV profile has a non-positive peak ({peak})")
    return peak
