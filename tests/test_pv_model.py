"""The PV chain replication, and the cross-year scale it exists to fix.

``flexcompute.pv_model`` re-implements upstream's pvlib model chain so that the
per-year normalisation upstream applies can be undone. That re-implementation
is a liability: if it ever drifts from upstream's, every solar number in the
project shifts silently and no other test would notice.

So the first test here is the important one -- our chain must reproduce
upstream's normalised profile **bit-for-bit**. The rest establish that undoing
the normalisation does what it is supposed to do, and that the TMY baseline is
untouched by it.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.pv_model import (
    unnormalised_dc_profile,
    upstream_normalisation_divisor,
)
from flexcompute.scenario import Scenario
from flexcompute.weather import get_tmy, get_weather_year, _historical_cache_paths

from conftest import BASELINE_SCENARIO, requires_weather

DALLAS = (32.78, -96.80)
PROBE_YEAR = 2019


def _year_cached(year: int) -> bool:
    data, meta = _historical_cache_paths("open_meteo_era5", *DALLAS, year)
    return data.exists() and meta.exists()


requires_probe_year = pytest.mark.skipif(
    not _year_cached(PROBE_YEAR),
    reason="run scripts/fetch_weather_years.py to cache historical years",
)


@requires_weather
def test_our_chain_reproduces_upstreams_profile_bit_for_bit():
    """The load-bearing test for this whole module.

    Normalise our replication the way upstream normalises its own, and the two
    must agree exactly. A failure here means upstream's PV configuration
    changed and every solar figure in the project needs re-deriving.
    """
    from flexcompute.upstream_bridge import ensure_upstream_importable

    ensure_upstream_importable()
    from pvstoragesim import get_solar_generation  # type: ignore[import-not-found]

    weather = get_tmy(*DALLAS, "pvgis")
    ours = unnormalised_dc_profile(weather.data, *DALLAS)
    ours_normalised = ours / ours.max()

    class _Load:
        tmy_weather = weather.data

    theirs = get_solar_generation(*DALLAS, facility_load=_Load())["p_dc"].to_numpy()

    assert np.array_equal(ours_normalised, theirs)


@requires_weather
def test_the_pvgis_tmy_divisor_is_exactly_one():
    """Why correcting the scale leaves the committed baseline alone.

    pvlib's PVWatts inverter clips AC output at ``pdc0 = 1``. The PVGIS TMY
    exceeds nameplate in some hours, so its maximum saturates at exactly 1.0
    and upstream's division is a no-op. This is asserted rather than assumed,
    because the baseline invariant depends on it.
    """
    divisor = upstream_normalisation_divisor(get_tmy(*DALLAS, "pvgis").data, *DALLAS)
    assert divisor == 1.0


@requires_weather
def test_site_solar_profile_is_unchanged_for_the_typical_year(site):
    """The correction must not move the reference scenario at all."""
    assert site.solar_p_dc.max() == pytest.approx(1.0, abs=0.0)


@requires_probe_year
def test_a_reanalysis_year_never_reaches_nameplate():
    """The mechanism behind the artefact, stated as a fact about the data.

    ERA5 smooths away the extreme clear-cold hours that push a real array past
    nameplate, so its annual peak sits below 1.0 and upstream's per-year
    division inflates the whole series.
    """
    weather = get_weather_year(*DALLAS, PROBE_YEAR, "open_meteo_era5")
    divisor = upstream_normalisation_divisor(weather.data, *DALLAS)
    assert 0.85 < divisor < 1.0


@requires_probe_year
def test_correction_restores_the_physical_scale():
    """After correction the profile peaks below nameplate, as it should."""
    site = Scenario(weather_year=PROBE_YEAR).build()
    peak = float(site.solar_p_dc.max())
    assert peak < 1.0
    weather = get_weather_year(*DALLAS, PROBE_YEAR, "open_meteo_era5")
    assert peak == pytest.approx(
        upstream_normalisation_divisor(weather.data, *DALLAS), rel=1e-9
    )


@requires_probe_year
def test_uncorrected_scale_would_have_inflated_the_year():
    """Quantifies what the correction removes, so it cannot be dismissed.

    Upstream's normalisation would have reported this year's capacity factor
    several percent higher than the physics supports.
    """
    site = Scenario(weather_year=PROBE_YEAR).build()
    corrected_cf = float(site.solar_p_dc.mean())
    weather = get_weather_year(*DALLAS, PROBE_YEAR, "open_meteo_era5")
    divisor = upstream_normalisation_divisor(weather.data, *DALLAS)
    uncorrected_cf = corrected_cf / divisor

    inflation = uncorrected_cf / corrected_cf - 1.0
    assert inflation > 0.01, "expected upstream's normalisation to inflate this year"


@pytest.mark.skipif(
    not all(_year_cached(y) for y in (2011, 2012, 2015)),
    reason="run scripts/fetch_weather_years.py to cache historical years",
)
def test_correction_changes_which_year_looks_sunniest():
    """The reason this matters for a multi-year study.

    Under per-year normalisation the divisor is smallest for the cloudiest
    years, which inflates them most and can lift a mediocre year above a good
    one. The corrected ranking is the physical one.
    """
    corrected, uncorrected = {}, {}
    for year in (2011, 2012, 2015):
        site = Scenario(weather_year=year).build()
        weather = get_weather_year(*DALLAS, year, "open_meteo_era5")
        divisor = upstream_normalisation_divisor(weather.data, *DALLAS)
        corrected[year] = float(site.solar_p_dc.mean())
        uncorrected[year] = corrected[year] / divisor

    assert max(corrected, key=corrected.get) == 2011
    # 2012 has the smallest divisor in the record and overtakes 2011 without
    # the correction: the artefact is rank-changing, not merely a level shift.
    assert max(uncorrected, key=uncorrected.get) == 2012
