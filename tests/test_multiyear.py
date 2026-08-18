"""Multi-year study machinery: year sets, droughts, aggregation, calibration.

The claims this study will make are claims about a *distribution* over weather
years, so the machinery that produces that distribution has to be as trustworthy
as the simulator underneath it. Three things are defended here:

* a year set is one source, in order, with no duplicates — because a source
  change between two years is indistinguishable from weather in the output;
* a drought search finds the real worst window, including one that straddles
  New Year, which a naive scan would cut in half;
* "10% forecast error" means the same realised error in every year, rather than
  the same internal sigma landing at different errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.forecast import (
    NRMSE_SWEEP_PCT,
    REFERENCE_NRMSE_24H_PCT,
    NoisySolarForecast,
    calibrate_sigma_for_nrmse,
    forecast_at_realised_nrmse,
)
from flexcompute.multiyear import (
    DALLAS_ERA5,
    DALLAS_STUDY_YEARS,
    DROUGHT_WINDOWS_H,
    Job,
    YearSet,
    aggregate,
    aggregate_over_years,
    concentration_of_advantage,
    drought_profile,
    worst_solar_window,
    year_summary,
)
from flexcompute.scenario import Scenario
from flexcompute.weather import _historical_cache_paths

from conftest import requires_weather

DALLAS = (32.78, -96.80)


def _cached(year: int) -> bool:
    data, meta = _historical_cache_paths("open_meteo_era5", *DALLAS, year)
    return data.exists() and meta.exists()


requires_years = pytest.mark.skipif(
    not all(_cached(y) for y in DALLAS_STUDY_YEARS),
    reason="run scripts/fetch_weather_years.py to cache the study years",
)


# ---------------------------------------------------------------------------
# Year sets
# ---------------------------------------------------------------------------

def test_the_study_window_is_fifteen_consecutive_years():
    assert DALLAS_STUDY_YEARS == tuple(range(2010, 2025))
    assert len(DALLAS_STUDY_YEARS) == 15


def test_year_set_rejects_duplicates_and_disorder():
    with pytest.raises(ValueError, match="duplicates"):
        YearSet(years=(2011, 2011))
    with pytest.raises(ValueError, match="ascending"):
        YearSet(years=(2012, 2011))


def test_year_set_builds_one_scenario_per_year_from_one_source():
    scenarios = DALLAS_ERA5.scenarios()
    assert len(scenarios) == 15
    assert {s.historical_weather_source for s in scenarios} == {"open_meteo_era5"}
    assert [s.weather_year for s in scenarios] == list(DALLAS_STUDY_YEARS)
    assert all(s.uses_historical_weather for s in scenarios)


def test_year_set_metadata_records_the_no_averaging_rule():
    meta = DALLAS_ERA5.metadata()
    assert meta["n_years"] == 15
    assert "never averaged" in meta["note"].lower()


# ---------------------------------------------------------------------------
# Droughts
# ---------------------------------------------------------------------------

class _FakeIndex:
    def __init__(self, n): self.n = n
    def __getitem__(self, i):
        class _T:
            @staticmethod
            def strftime(fmt): return f"hour{i}"
        return _T


class _FakeSite:
    """Minimal stand-in: the drought search only reads solar and the index."""

    def __init__(self, profile):
        self.solar_p_dc = np.asarray(profile, dtype=float)

        class _TMY:
            index = _FakeIndex(len(profile))
        class _Wrap:
            data = _TMY()
        self.tmy = _Wrap()


def test_worst_window_finds_the_obvious_hole():
    profile = np.full(8760, 0.5)
    profile[3000:3072] = 0.0
    drought = worst_solar_window(_FakeSite(profile), 72)
    assert drought.start_hour == 3000
    assert drought.mean_solar_fraction == pytest.approx(0.0)


def test_worst_window_wraps_across_new_year():
    """A drought straddling 31 December is one event, not two halves.

    A naive scan would report only the part that fits inside the calendar and
    understate the worst event in the record. The simulator already treats the
    year as a loop through the cyclic SOC condition, so wrapping is consistent
    with how the year is actually run.
    """
    profile = np.full(8760, 0.5)
    profile[8724:] = 0.0        # last 36 hours
    profile[:36] = 0.0          # first 36 hours
    drought = worst_solar_window(_FakeSite(profile), 72)
    assert drought.start_hour == 8724
    assert drought.mean_solar_fraction == pytest.approx(0.0)


def test_severity_is_relative_to_the_years_own_mean():
    profile = np.full(8760, 0.4)
    profile[100:172] = 0.1
    drought = worst_solar_window(_FakeSite(profile), 72)
    assert drought.severity < 1.0
    assert drought.severity == pytest.approx(
        drought.mean_solar_fraction / drought.annual_mean_fraction
    )


def test_window_length_must_be_valid():
    site = _FakeSite(np.full(8760, 0.3))
    with pytest.raises(ValueError):
        worst_solar_window(site, 0)
    with pytest.raises(ValueError):
        worst_solar_window(site, 9000)


def test_longer_windows_are_never_more_severe_in_absolute_terms():
    """A one-week mean cannot be lower than the worst single day inside it."""
    rng = np.random.default_rng(0)
    site = _FakeSite(np.clip(rng.normal(0.3, 0.2, 8760), 0, 1))
    profile = drought_profile(site)
    means = [profile[w].mean_solar_fraction for w in sorted(profile)]
    assert all(a <= b + 1e-12 for a, b in zip(means, means[1:]))


@requires_weather
def test_drought_profile_covers_every_requested_window(site):
    profile = drought_profile(site)
    assert set(profile) == set(DROUGHT_WINDOWS_H)


@pytest.mark.skipif(not _cached(2015), reason="cache historical years first")
def test_the_hard_year_has_a_genuinely_hard_week():
    """2015 is the drought year in the Dallas record; pin that it stays so.

    Not a tuning knob — a fact about the weather. If this moves, either the
    weather pipeline changed or the source did.
    """
    site = Scenario(weather_year=2015).build()
    worst = worst_solar_window(site, 72)
    assert worst.mean_solar_fraction < 0.02
    assert worst.severity < 0.10


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_aggregate_reports_level_and_spread_together():
    stats = aggregate(range(1, 16))
    assert stats["n"] == 15
    assert stats["mean"] == pytest.approx(8.0)
    assert stats["median"] == pytest.approx(8.0)
    assert stats["min"] == 1.0 and stats["max"] == 15.0
    assert stats["p10"] < stats["median"] < stats["p90"]


def test_aggregate_survives_an_empty_or_nan_sample():
    assert aggregate([])["n"] == 0
    assert aggregate([float("nan")])["n"] == 0
    assert aggregate([1.0, float("nan"), 3.0])["n"] == 2


def test_aggregate_over_years_picks_out_named_metrics():
    per_year = {2010: {"compute_units": 10.0}, 2011: {"compute_units": 20.0}}
    out = aggregate_over_years(per_year, ["compute_units"])
    assert out["compute_units"]["mean"] == pytest.approx(15.0)


def test_concentration_detects_a_result_carried_by_a_few_years():
    """The statistic that answers 'consistent, or two bad years?'."""
    carried = {y: (100.0 if y < 2012 else 0.1) for y in DALLAS_STUDY_YEARS}
    even = {y: 10.0 for y in DALLAS_STUDY_YEARS}

    assert concentration_of_advantage(carried)["top_2_year_share"] > 0.9
    assert concentration_of_advantage(even)["top_3_year_share"] == pytest.approx(0.2)
    assert concentration_of_advantage(even)["even_split_top_3_share"] == pytest.approx(0.2)


def test_concentration_counts_years_that_actually_gained():
    mixed = {y: (5.0 if y % 2 == 0 else -1.0) for y in DALLAS_STUDY_YEARS}
    out = concentration_of_advantage(mixed)
    assert out["years_evaluated"] == 15
    assert out["years_with_positive_advantage"] == sum(
        1 for y in DALLAS_STUDY_YEARS if y % 2 == 0
    )


def test_concentration_shares_stay_bounded_when_some_years_lose():
    """Shares are taken against gross gain, not the net.

    Dividing by a net total that mixes gains and losses can exceed 100% or
    flip sign — a number that reads like an answer and is not one.
    """
    mixed = {2010: 10.0, 2011: 10.0, 2012: -18.0, 2013: 1.0}
    out = concentration_of_advantage(mixed)
    assert out["net_advantage"] == pytest.approx(3.0)
    assert out["gross_gain"] == pytest.approx(21.0)
    for k in ("top_1_year_share", "top_2_year_share", "top_3_year_share"):
        assert 0.0 <= out[k] <= 1.0
    assert out["top_3_year_share"] == pytest.approx(1.0)


def test_concentration_is_undefined_when_nothing_gained():
    """A strategy that loses every year has no advantage to apportion."""
    out = concentration_of_advantage({y: -1.0 for y in DALLAS_STUDY_YEARS})
    assert out["interpretable"] is False
    assert np.isnan(out["top_1_year_share"])
    assert out["years_with_positive_advantage"] == 0


def test_even_split_reference_matches_the_sample_size():
    """With fewer than three years the top-3 share is trivially everything."""
    assert concentration_of_advantage({1: 1.0, 2: 1.0})["even_split_top_3_share"] == 1.0
    assert concentration_of_advantage(
        {y: 1.0 for y in DALLAS_STUDY_YEARS}
    )["even_split_top_3_share"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Forecast calibration
# ---------------------------------------------------------------------------

@requires_weather
def test_calibration_hits_the_requested_realised_error(site):
    solar = site.solar_p_dc * 80.0
    for target in NRMSE_SWEEP_PCT:
        forecast = forecast_at_realised_nrmse(solar, target)
        assert forecast.realised_nrmse_24h_pct() == pytest.approx(target, abs=0.05)


@requires_weather
def test_calibration_is_monotone_in_sigma(site):
    solar = site.solar_p_dc * 80.0
    sigmas = [calibrate_sigma_for_nrmse(solar, t) for t in NRMSE_SWEEP_PCT]
    assert all(a < b for a, b in zip(sigmas, sigmas[1:]))


@pytest.mark.skipif(
    not (_cached(2015) and _cached(2019)), reason="cache historical years first"
)
def test_the_same_sigma_is_a_different_error_in_different_years():
    """Why the sweep is indexed by realised error and not by sigma.

    A sweep labelled by sigma would silently compare different error levels
    across years and call them the same level.
    """
    solars = {
        y: Scenario(weather_year=y).build().solar_p_dc * 80.0 for y in (2015, 2019)
    }
    fixed_sigma = {
        y: NoisySolarForecast(s, sigma_24h=0.15).realised_nrmse_24h_pct()
        for y, s in solars.items()
    }
    calibrated = {
        y: forecast_at_realised_nrmse(s, REFERENCE_NRMSE_24H_PCT).realised_nrmse_24h_pct()
        for y, s in solars.items()
    }
    assert abs(fixed_sigma[2015] - fixed_sigma[2019]) > 0.2
    assert abs(calibrated[2015] - calibrated[2019]) < 0.05


def test_reference_error_is_a_magnitude_not_a_frequency():
    """Guards the framing, which has been got wrong before.

    10% nRMSE is an error magnitude normalised by plant capacity. It is not
    "wrong 10% of the time", and nothing in the code should imply that.
    """
    assert REFERENCE_NRMSE_24H_PCT == 10.0
    assert 10.0 in NRMSE_SWEEP_PCT
    assert min(NRMSE_SWEEP_PCT) == 5.0 and max(NRMSE_SWEEP_PCT) == 20.0


def test_calibration_rejects_a_nonsense_target():
    with pytest.raises(ValueError):
        calibrate_sigma_for_nrmse(np.ones(8760), 0.0)


# ---------------------------------------------------------------------------
# Parallel execution plumbing
# ---------------------------------------------------------------------------

@requires_weather
def test_a_job_round_trips_through_the_serial_path():
    from flexcompute.multiyear import run_jobs

    job = Job(
        scenario=Scenario(),
        strategy="fixed_load",
        solar_mw=120.0, battery_mw=60.0, duration_h=6.0,
    )
    out = run_jobs([job], workers=1)
    key, metrics = next(iter(out.items()))
    assert key == (None, "fixed_load")
    assert metrics["_strategy"] == "fixed_load"
    assert metrics["compute_units"] > 0
    # Provenance that must travel with every row of a multi-year table.
    assert metrics["_curve"] and metrics["_curve_kind"]
    assert metrics["_sizing"]["solar_mw_dc"] == pytest.approx(120.0)


@pytest.mark.skipif(not _cached(2019), reason="cache historical years first")
def test_parallel_and_serial_paths_agree():
    from flexcompute.multiyear import run_jobs

    jobs = [
        Job(scenario=Scenario(weather_year=y), strategy=s,
            solar_mw=80.0, battery_mw=40.0, duration_h=6.0)
        for y in (2019,) for s in ("fixed_load", "casey_governor")
    ]
    serial = run_jobs(jobs, workers=1)
    parallel = run_jobs(jobs, workers=2)
    assert set(serial) == set(parallel)
    for key in serial:
        assert serial[key]["compute_units"] == pytest.approx(
            parallel[key]["compute_units"], rel=1e-12
        )
