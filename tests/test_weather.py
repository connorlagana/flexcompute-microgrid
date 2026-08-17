"""Weather normalisation tests.

The hour-of-year index is load-bearing: PUE, the IT load shape, and the solar
profile are all positional 8760 arrays that are assumed to line up. If the
local-time normalisation is wrong, every downstream number is quietly wrong
and nothing crashes.
"""

from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from flexcompute.weather import (
    HOURS_PER_YEAR,
    REFERENCE_YEAR,
    REQUIRED_COLUMNS,
    to_local_standard_hour_index,
)

from conftest import requires_weather


def _utc_frame(year: int = 1990) -> pd.DataFrame:
    """A synthetic UTC-indexed year whose values encode their own UTC hour."""
    idx = pd.date_range(f"{year}-01-01", periods=HOURS_PER_YEAR, freq="h", tz="UTC")
    return pd.DataFrame({col: np.arange(HOURS_PER_YEAR, dtype=float) for col in REQUIRED_COLUMNS},
                        index=idx)


def test_utc_input_is_rotated_to_local_standard_time():
    """Row i of the output must be the UTC instant that is local hour i."""
    offset = -6.0  # US Central Standard Time
    out = to_local_standard_hour_index(_utc_frame(), offset)

    assert len(out) == HOURS_PER_YEAR
    assert out.index[0] == pd.Timestamp(f"{REFERENCE_YEAR}-01-01 00:00", tz=timezone(timedelta(hours=offset)))

    # Local hour j corresponds to UTC hour j - offset = j + 6, wrapping the year.
    values = out["ghi"].to_numpy()
    expected = (np.arange(HOURS_PER_YEAR) + 6) % HOURS_PER_YEAR
    np.testing.assert_array_equal(values, expected)


def test_index_is_contiguous_hourly_local_time():
    out = to_local_standard_hour_index(_utc_frame(), -6.0)
    deltas = np.unique(np.diff(out.index.to_numpy()))
    assert len(deltas) == 1
    assert deltas[0] == np.timedelta64(1, "h")


def test_zero_offset_is_identity_on_ordering():
    out = to_local_standard_hour_index(_utc_frame(), 0.0)
    np.testing.assert_array_equal(out["ghi"].to_numpy(), np.arange(HOURS_PER_YEAR))


def test_leap_day_rows_are_dropped():
    """A TMY month drawn from a leap year must not produce 8784 rows."""
    idx = pd.date_range("1992-01-01", periods=8784, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {col: np.arange(8784, dtype=float) for col in REQUIRED_COLUMNS}, index=idx
    )
    out = to_local_standard_hour_index(frame, 0.0)
    assert len(out) == HOURS_PER_YEAR


def test_wrong_length_is_rejected():
    idx = pd.date_range("1990-01-01", periods=100, freq="h", tz="UTC")
    frame = pd.DataFrame({col: np.zeros(100) for col in REQUIRED_COLUMNS}, index=idx)
    with pytest.raises(ValueError, match="8760"):
        to_local_standard_hour_index(frame, 0.0)


# ---------------------------------------------------------------------------
# The real site
# ---------------------------------------------------------------------------

@requires_weather
def test_dallas_solar_peaks_around_local_noon(site):
    """End-to-end alignment check on real data.

    If the timezone handling were wrong, peak irradiance would land in the
    small hours and nothing else would complain.
    """
    ghi = site.tmy.data["ghi"].to_numpy()
    hour_of_day = np.arange(HOURS_PER_YEAR) % 24
    by_hour = np.array([ghi[hour_of_day == h].mean() for h in range(24)])
    assert 10 <= int(by_hour.argmax()) <= 13
    assert by_hour[0] == 0.0  # local midnight
    assert by_hour[23] == 0.0


@requires_weather
def test_solar_profile_is_dark_at_night(site):
    """PV production must be zero overnight, in the same index convention."""
    p_dc = site.solar_p_dc
    hour_of_day = np.arange(HOURS_PER_YEAR) % 24
    night = np.isin(hour_of_day, [0, 1, 2, 3, 22, 23])
    assert p_dc[night].max() == pytest.approx(0.0, abs=1e-9)
    assert p_dc.max() == pytest.approx(1.0)


@requires_weather
def test_all_site_arrays_share_one_length(site):
    assert site.it_load_mw.shape == (HOURS_PER_YEAR,)
    assert site.hourly_pue.shape == (HOURS_PER_YEAR,)
    assert site.solar_p_dc.shape == (HOURS_PER_YEAR,)
    assert len(site.tmy.data) == HOURS_PER_YEAR
