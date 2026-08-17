"""Forecast tests: the belief a controller is allowed to hold.

The whole project hinges on one separation -- ground truth is what the
dispatcher simulates, a forecast is what the controller believes -- so these
tests are mostly about keeping the two apart and making the error structurally
honest.

Three failure modes are worth naming, because each would silently inflate the
project's headline number:

1. **A forecast that is accidentally perfect.** Error expressed as a fraction of
   *realised* output is exactly zero during a drought, which is precisely when
   control matters. Tested directly against real weather.
2. **Error that averages away.** White noise cancels inside a 48-hour plan, so
   an MPC barely notices it. The error has to persist on weather timescales.
3. **A perfect-foresight run wearing a noisy label, or the reverse.** Guarded by
   a type check, tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.forecast import (
    NoisySolarForecast,
    PerfectSolarForecast,
    SolarForecast,
    clear_sky_envelope,
)

from conftest import BASELINE_SCENARIO, requires_weather

HOURS = 8760


@pytest.fixture(scope="module")
def synthetic_year() -> np.ndarray:
    """A year with a diurnal shape, a season and day-to-day cloudiness."""
    rng = np.random.default_rng(0)
    h = np.arange(HOURS)
    day = np.clip(np.sin((h % 24 - 6) / 12 * np.pi), 0.0, None)
    season = 0.75 + 0.25 * np.sin(h / HOURS * 2 * np.pi - np.pi / 2)
    cloud = np.repeat(rng.uniform(0.05, 1.0, 365), 24)
    return 200.0 * day * season * cloud


@pytest.fixture(scope="module")
def real_solar() -> np.ndarray:
    """The realised DC series the experiments actually use."""
    site = BASELINE_SCENARIO.build()
    return site.solar_p_dc * 200.0


# ---------------------------------------------------------------------------
# The clear-sky envelope
# ---------------------------------------------------------------------------

def test_envelope_bounds_the_truth_everywhere(synthetic_year):
    envelope = clear_sky_envelope(synthetic_year)
    assert np.all(envelope >= synthetic_year - 1e-12)


def test_envelope_is_exactly_zero_at_night(synthetic_year):
    """Nobody mis-forecasts darkness, so the error model must vanish there."""
    envelope = clear_sky_envelope(synthetic_year)
    night = synthetic_year.reshape(365, 24)[:, :4].reshape(-1)
    assert night.max() == 0.0
    assert envelope.reshape(365, 24)[:, :4].max() == 0.0


def test_envelope_follows_the_season(synthetic_year):
    """It is climatology, not a constant: summer noon must exceed winter noon."""
    envelope = clear_sky_envelope(synthetic_year).reshape(365, 24)
    assert envelope[172, 12] > envelope[355, 12]


def test_envelope_handles_a_partial_final_day():
    truth = np.linspace(0.0, 10.0, 8761)
    assert len(clear_sky_envelope(truth)) == 8761


def test_envelope_rejects_a_degenerate_window():
    with pytest.raises(ValueError):
        clear_sky_envelope(np.zeros(48), window_days=0)


# ---------------------------------------------------------------------------
# The error model
# ---------------------------------------------------------------------------

def test_noisy_forecast_satisfies_the_protocol(synthetic_year):
    assert isinstance(NoisySolarForecast(synthetic_year), SolarForecast)


def test_the_current_hour_is_known_exactly(synthetic_year):
    """Lead zero carries no error: the controller measures current irradiance.

    Bit-exact, not approximate -- ``Observation.solar_dc_mw`` hands the
    dispatcher's own value to every other controller, and a forecast-aware one
    must not be handicapped relative to them at t=0.
    """
    forecast = NoisySolarForecast(synthetic_year, sigma_24h=0.30, seed=5)
    for t in (0, 137, 4000, 8759):
        assert forecast.horizon(t, 48)[0] == synthetic_year[t]


def test_the_same_question_gets_the_same_answer(synthetic_year):
    """A forecast is a pure function of (t, hours, seed). Runs must reproduce."""
    forecast = NoisySolarForecast(synthetic_year, seed=11)
    np.testing.assert_array_equal(forecast.horizon(500, 48), forecast.horizon(500, 48))
    np.testing.assert_array_equal(
        forecast.horizon(500, 96)[:48], forecast.horizon(500, 48)
    )
    twin = NoisySolarForecast(synthetic_year, seed=11)
    np.testing.assert_array_equal(forecast.horizon(500, 48), twin.horizon(500, 48))


def test_different_seeds_are_different_realisations(synthetic_year):
    a = NoisySolarForecast(synthetic_year, seed=1).horizon(500, 48)
    b = NoisySolarForecast(synthetic_year, seed=2).horizon(500, 48)
    assert not np.array_equal(a, b)


def test_zero_sigma_collapses_to_perfect_foresight(synthetic_year):
    """The error model must contain the perfect case exactly, not nearly.

    This is what makes sigma a genuine dial between the two experiments rather
    than a different controller.
    """
    forecast = NoisySolarForecast(synthetic_year, sigma_24h=0.0)
    np.testing.assert_array_equal(forecast.horizon(0, HOURS), synthetic_year)
    np.testing.assert_array_equal(
        forecast.horizon(3000, 72), PerfectSolarForecast(synthetic_year).horizon(3000, 72)
    )


def test_error_grows_with_lead_time(synthetic_year):
    stats = NoisySolarForecast(synthetic_year, seed=3).error_stats((1, 6, 12, 24, 48))
    rmse = [stats[f"lead_{k}h"]["rmse_mw"] for k in (1, 6, 12, 24, 48)]
    assert rmse == sorted(rmse)
    assert rmse[0] > 0.0


def test_error_saturates_beyond_the_saturation_horizon(synthetic_year):
    """Past a few days a forecast is climatology and stops getting worse."""
    forecast = NoisySolarForecast(synthetic_year, saturation_hours=72.0, seed=3)
    stats = forecast.error_stats((72, 168))
    assert stats["lead_72h"]["rmse_mw"] == pytest.approx(stats["lead_168h"]["rmse_mw"])


def test_error_scales_with_sigma(synthetic_year):
    def rmse(sigma: float) -> float:
        return NoisySolarForecast(synthetic_year, sigma_24h=sigma, seed=3).error_stats(
            (24,)
        )["lead_24h"]["rmse_mw"]

    assert rmse(0.30) > rmse(0.15) > rmse(0.05) > 0.0


def test_the_forecast_stays_physical(synthetic_year):
    """Never negative, never above what a clear sky could deliver."""
    forecast = NoisySolarForecast(synthetic_year, sigma_24h=0.30, bias=0.2, seed=9)
    envelope = clear_sky_envelope(synthetic_year)
    believed = forecast.forecast_at_lead(48)
    assert believed.min() >= 0.0
    assert np.all(believed <= envelope + 1e-9)


def test_errors_persist_rather_than_cancelling(synthetic_year):
    """Correlated error, not white noise -- the distinction an MPC feels.

    Independent hourly errors average out inside a 48-hour plan, so a controller
    would barely notice them and the experiment would flatter itself. Checked as
    autocorrelation of the daylight error at a 6-hour lag.
    """
    forecast = NoisySolarForecast(synthetic_year, correlation_hours=24.0, seed=4)
    envelope = clear_sky_envelope(synthetic_year)
    error = (forecast.forecast_at_lead(24) - synthetic_year)[envelope > 0]
    error = error - error.mean()
    lag = 6
    autocorr = float(
        np.dot(error[:-lag], error[lag:]) / np.dot(error, error)
    )
    assert autocorr > 0.2

    white = NoisySolarForecast(synthetic_year, correlation_hours=0.0, seed=4)
    w_error = (white.forecast_at_lead(24) - synthetic_year)[envelope > 0]
    w_error = w_error - w_error.mean()
    w_autocorr = float(np.dot(w_error[:-lag], w_error[lag:]) / np.dot(w_error, w_error))
    assert autocorr > w_autocorr


def test_bias_shifts_the_forecast_in_one_direction(synthetic_year):
    envelope = clear_sky_envelope(synthetic_year) > 0
    optimistic = NoisySolarForecast(synthetic_year, bias=0.15, seed=6)
    pessimistic = NoisySolarForecast(synthetic_year, bias=-0.15, seed=6)
    assert optimistic.error_stats((24,))["lead_24h"]["bias_mw"] > 0.0
    assert pessimistic.error_stats((24,))["lead_24h"]["bias_mw"] < 0.0
    assert np.mean(optimistic.forecast_at_lead(24)[envelope]) > np.mean(
        pessimistic.forecast_at_lead(24)[envelope]
    )


def test_horizon_truncates_and_never_wraps(synthetic_year):
    forecast = NoisySolarForecast(synthetic_year)
    assert len(forecast.horizon(HOURS - 10, 48)) == 10
    assert len(forecast.horizon(HOURS, 48)) == 0
    assert len(forecast.horizon(100, 0)) == 0


def test_metadata_carries_the_error_level_and_its_caveat(synthetic_year):
    """No number from a noisy run is quotable without knowing how noisy."""
    meta = NoisySolarForecast(synthetic_year, sigma_24h=0.2, seed=8).metadata()
    assert meta["kind"] == "noisy"
    assert meta["parameters"]["sigma_24h"] == 0.2
    assert meta["parameters"]["seed"] == 8
    assert meta["realised_error"]["lead_24h"]["nrmse_pct_of_capacity"] > 0.0
    assert "B13" in meta["caveat"]


def test_rejects_an_envelope_that_is_not_an_upper_bound(synthetic_year):
    with pytest.raises(ValueError, match="upper bound"):
        NoisySolarForecast(synthetic_year, envelope_dc_mw=synthetic_year * 0.5)


def test_rejects_a_negative_sigma(synthetic_year):
    with pytest.raises(ValueError):
        NoisySolarForecast(synthetic_year, sigma_24h=-0.1)


# ---------------------------------------------------------------------------
# The failure mode that decides whether any of this is honest
# ---------------------------------------------------------------------------

@requires_weather
def test_the_forecast_can_miss_a_real_drought(real_solar):
    """It must be able to promise sun on a day that turns out overcast.

    This is the whole reason error is scaled by the clear-sky envelope rather
    than by realised output. A multiplicative-on-truth model predicts zero
    output on a dark day with zero error -- it would forecast every multi-day
    drought perfectly, which are exactly the events that size an islanded plant
    and exactly where a controller can be hurt. If this test fails the error
    model is decorative.

    Checked against real PVGIS weather rather than a synthetic year, because
    synthetic cloudiness rarely produces a genuinely dark daylight hour.
    """
    forecast = NoisySolarForecast(real_solar, sigma_24h=0.15, seed=20260815)
    envelope = clear_sky_envelope(real_solar)
    believed = forecast.forecast_at_lead(24)

    # Hours the sun should have been well up, but was not.
    bright = envelope > 0.30 * envelope.max()
    overcast = bright & (real_solar < 0.25 * envelope)
    assert overcast.sum() > 100, "no real overcast daylight hours to test against"

    over_promised = overcast & (believed > real_solar + 0.20 * envelope)
    assert over_promised.sum() > 0, (
        "the forecast never over-promised during an overcast hour; error is "
        "structurally unable to miss a drought"
    )

    # And the converse: it must also be able to under-promise on a clear day.
    clear = bright & (real_solar > 0.85 * envelope)
    under_promised = clear & (believed < real_solar - 0.20 * envelope)
    assert under_promised.sum() > 0


@requires_weather
def test_realised_skill_is_in_a_plausible_day_ahead_range(real_solar):
    """Sanity-check the dial against what day-ahead forecasting achieves.

    ``sigma_24h`` is a fraction of each hour's clear-sky potential, so it does
    not equal published nRMSE. This pins the translation: the default level must
    land in the neighbourhood of a real single-site day-ahead forecast rather
    than being either trivially easy or absurdly bad. It is a range check, not a
    validation against any specific published dataset -- see ASSUMPTIONS B13.
    """
    stats = NoisySolarForecast(real_solar, sigma_24h=0.15, seed=20260815).error_stats(
        (24,)
    )["lead_24h"]
    assert 3.0 < stats["nrmse_pct_of_capacity"] < 20.0
