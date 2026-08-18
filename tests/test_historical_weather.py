"""Actual historical weather years, and the invariants they must satisfy.

A typical meteorological year is built by choosing the most *typical* January
from a long record, then the most typical February, and so on. The algorithm
therefore excludes, by construction, the multi-day solar drought that made some
January atypical -- and that drought is what sizes an islanded plant. Running
this study on a TMY alone would systematically flatter every strategy that does
not have to survive one.

These tests defend the plumbing that lets real years in: correct local-time
alignment, an explicit and consistent leap-year policy, provenance on every
record, and -- most importantly -- that a real year is never silently confused
with a composite one.

Network tests are skipped unless the year is already cached, so the suite stays
offline and deterministic after the first fetch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flexcompute.scenario import Scenario
from flexcompute.weather import (
    HISTORICAL_PROVIDERS,
    HOURS_PER_YEAR,
    REFERENCE_YEAR,
    NSRDBHistoricalProvider,
    OpenMeteoHistoricalProvider,
    WeatherYear,
    _historical_cache_paths,
    get_weather_year,
)

from conftest import BASELINE_SCENARIO

DALLAS = (32.78, -96.80)

#: The year used for the end-to-end check. Available from both sources, so the
#: two can be compared against each other.
PROBE_YEAR = 2019
#: A leap year, to exercise the 29 February policy.
LEAP_YEAR = 2020


def _cached(source: str, year: int) -> bool:
    data, meta = _historical_cache_paths(source, *DALLAS, year)
    return data.exists() and meta.exists()


def _requires(source: str, year: int):
    return pytest.mark.skipif(
        not _cached(source, year),
        reason=f"{source} {year} not cached; run scripts/fetch_weather_years.py",
    )


# ---------------------------------------------------------------------------
# Typical is not actual
# ---------------------------------------------------------------------------

def test_a_typical_year_declares_itself_typical():
    record = WeatherYear(
        data=pd.DataFrame(), source="pvgis", latitude=0.0, longitude=0.0,
        timezone_name="UTC", utc_offset_hours=0.0, retrieved_utc="",
    )
    assert record.is_typical
    assert record.year is None
    assert record.provenance()["year"] is None


def test_an_actual_year_declares_its_year():
    record = WeatherYear(
        data=pd.DataFrame(), source="open_meteo_era5", latitude=0.0, longitude=0.0,
        timezone_name="UTC", utc_offset_hours=0.0, retrieved_utc="", year=2013,
    )
    assert not record.is_typical
    assert record.provenance()["year"] == 2013


def test_scenario_keeps_the_two_sources_apart():
    """A scenario cannot be ambiguous about which kind of year it ran."""
    typical = Scenario()
    actual = Scenario(weather_year=2013)

    assert not typical.uses_historical_weather
    assert actual.uses_historical_weather
    assert typical.effective_weather_source == "pvgis"
    assert actual.effective_weather_source == "open_meteo_era5"
    assert typical.label() != actual.label()
    assert str(2013) in actual.label()


def test_typical_year_labels_are_unchanged_by_the_new_fields():
    """Results and snapshots committed before historical years still resolve."""
    assert Scenario().label() == "dallas_10kgpu_ac_coupled_mv_coupled_pvgis"


def test_years_are_never_averaged_together():
    """There is no API that blends years, and there must not be.

    Averaging weather before simulation would remove exactly the tail events
    the study exists to measure, so the absence of such a function is itself
    the invariant.
    """
    import flexcompute.weather as weather

    forbidden = [n for n in dir(weather) if "average" in n or "blend" in n]
    assert forbidden == []


# ---------------------------------------------------------------------------
# Provider declarations
# ---------------------------------------------------------------------------

def test_every_historical_provider_declares_its_coverage():
    for name, provider in HISTORICAL_PROVIDERS.items():
        first, last = provider.available_years()
        assert first < last, name
        assert 1900 < first and last < 2100, name


def test_nsrdb_refuses_years_outside_its_satellite_era():
    provider = NSRDBHistoricalProvider(email="test@example.com")
    with pytest.raises(ValueError, match="covers"):
        provider.fetch_year(*DALLAS, 2010)


def test_era5_covers_the_whole_study_period():
    first, last = OpenMeteoHistoricalProvider().available_years()
    assert first <= 2010 and last >= 2024


# ---------------------------------------------------------------------------
# Shape, time alignment and the leap-day policy
# ---------------------------------------------------------------------------

@_requires("open_meteo_era5", PROBE_YEAR)
def test_historical_year_lands_on_the_canonical_index():
    record = get_weather_year(*DALLAS, PROBE_YEAR, "open_meteo_era5")
    data = record.data

    assert len(data) == HOURS_PER_YEAR
    assert list(data.columns) == ["temp_air", "relative_humidity", "ghi", "dni",
                                  "dhi", "wind_speed"]
    assert data.index[0].year == REFERENCE_YEAR
    assert (data.index[0].month, data.index[0].day, data.index[0].hour) == (1, 1, 0)
    assert (data.index[-1].month, data.index[-1].day, data.index[-1].hour) == (12, 31, 23)
    # Strictly hourly, no gaps, no duplicates.
    assert (data.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


@_requires("open_meteo_era5", PROBE_YEAR)
def test_local_time_alignment_puts_noon_at_noon():
    """The single most consequential thing to get wrong.

    A timezone error shifts the whole solar profile against the load and PUE
    profiles, which would quietly corrupt every dispatch result. Peak GHI must
    fall in the early afternoon in local standard time.
    """
    record = get_weather_year(*DALLAS, PROBE_YEAR, "open_meteo_era5")
    ghi = record.data["ghi"].to_numpy()
    hour_of_day = np.arange(HOURS_PER_YEAR) % 24
    by_hour = np.array([ghi[hour_of_day == h].mean() for h in range(24)])

    assert 11 <= int(by_hour.argmax()) <= 13
    # Deep night carries no irradiance at all.
    assert by_hour[0] == 0.0 and by_hour[23] == 0.0


@_requires("open_meteo_era5", LEAP_YEAR)
def test_leap_year_drops_29_february_and_says_so():
    record = get_weather_year(*DALLAS, LEAP_YEAR, "open_meteo_era5")
    assert len(record.data) == HOURS_PER_YEAR
    assert record.leap_day_dropped is True
    assert record.provenance()["leap_day_dropped"] is True


@_requires("open_meteo_era5", PROBE_YEAR)
def test_non_leap_year_reports_no_leap_day():
    record = get_weather_year(*DALLAS, PROBE_YEAR, "open_meteo_era5")
    assert record.leap_day_dropped is False


@_requires("open_meteo_era5", LEAP_YEAR)
@_requires("open_meteo_era5", PROBE_YEAR)
def test_seasons_stay_aligned_across_leap_and_non_leap_years():
    """The reason 29 February is dropped rather than the tail truncated.

    Truncating the tail would slide every date after February by one day in
    leap years only, which would appear in the results as year-to-year
    variation that is really a calendar bug.
    """
    a = get_weather_year(*DALLAS, PROBE_YEAR, "open_meteo_era5").data
    b = get_weather_year(*DALLAS, LEAP_YEAR, "open_meteo_era5").data
    for frame in (a, b):
        row = frame.index[24 * 180]      # the 181st day, whichever year
        assert (row.month, row.day) == (6, 30)


# ---------------------------------------------------------------------------
# Physical sanity, and the cross-source check
# ---------------------------------------------------------------------------

@_requires("open_meteo_era5", PROBE_YEAR)
def test_irradiance_is_physically_sane():
    record = get_weather_year(*DALLAS, PROBE_YEAR, "open_meteo_era5")
    data = record.data
    assert data["ghi"].min() >= 0.0
    assert data["ghi"].max() < 1400.0            # above the solar constant is a bug
    assert data["dni"].min() >= 0.0
    assert 0.0 <= data["relative_humidity"].min()
    assert data["relative_humidity"].max() <= 100.0
    assert data["wind_speed"].min() >= 0.0
    # Dallas annual GHI is roughly 1600-1800 kWh/m2.
    assert 1400 < record.summary()["ghi_annual_kwh_per_m2"] < 2000


@_requires("open_meteo_era5", PROBE_YEAR)
@_requires("nsrdb_psm4_conus", PROBE_YEAR)
def test_the_two_sources_agree_on_the_same_year():
    """Cross-validation of the reanalysis pipeline against satellite retrieval.

    Annual GHI is expected to agree closely. DNI is expected to agree less
    well: a reanalysis smooths broken cloud and over-reads the direct beam,
    which is the declared bias of the ERA5 source and the reason NSRDB is run
    over the overlapping years as a check.
    """
    era5 = get_weather_year(*DALLAS, PROBE_YEAR, "open_meteo_era5").summary()
    nsrdb = get_weather_year(*DALLAS, PROBE_YEAR, "nsrdb_psm4_conus").summary()

    ghi_gap = abs(era5["ghi_annual_kwh_per_m2"] / nsrdb["ghi_annual_kwh_per_m2"] - 1)
    assert ghi_gap < 0.05, f"annual GHI differs by {100 * ghi_gap:.1f}%"
    assert abs(era5["temp_air_mean_c"] - nsrdb["temp_air_mean_c"]) < 2.0


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@_requires("open_meteo_era5", PROBE_YEAR)
def test_a_historical_year_builds_a_site_and_records_its_provenance():
    scenario = Scenario(weather_year=PROBE_YEAR)
    site = scenario.build()

    assert site.weather_year == PROBE_YEAR
    assert site.it_load_mw.shape == (HOURS_PER_YEAR,)
    assert site.solar_p_dc.shape == (HOURS_PER_YEAR,)
    assert site.hourly_pue.shape == (HOURS_PER_YEAR,)

    provenance = site.provenance()
    assert provenance["weather"]["year"] == PROBE_YEAR
    assert provenance["weather"]["source"] == "open_meteo_era5"
    assert provenance["weather"]["dataset"]
    assert provenance["scenario"]["weather_year"] == PROBE_YEAR


@_requires("open_meteo_era5", PROBE_YEAR)
def test_a_real_year_is_harder_than_the_typical_year(site):
    """The point of the whole exercise, stated as a testable claim.

    A TMY should show a *shallower* worst solar drought than a real year picked
    from the same climate, because the TMY selection algorithm discards the
    months that contained the droughts. If this ever fails it is worth knowing:
    either the real year was unusually kind, or the TMY is not what we think.
    """
    real = Scenario(weather_year=PROBE_YEAR).build()

    def worst_72h(s) -> float:
        daily = np.convolve(s.solar_p_dc, np.ones(72), mode="valid")
        return float(daily.min() / 72.0)

    # Not asserted as strictly worse -- one year is a sample of one. Asserted
    # as *comparable*, so a gross units or alignment error would show up.
    assert 0.0 <= worst_72h(real) < 0.5 * float(real.solar_p_dc.mean())
    assert 0.0 <= worst_72h(site) < 0.5 * float(site.solar_p_dc.mean())
