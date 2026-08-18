"""Solar geometry as a *clock*, not as a forecast.

A controller that rations stored energy overnight needs one number: **how long
until the sun comes back**. That number is astronomy — a function of latitude,
longitude and day-of-year — and it is knowable years in advance from an
ephemeris. It carries no information whatsoever about clouds.

Keeping it in its own module is the point. :mod:`flexcompute.forecast` supplies
*beliefs about weather*, which a controller may only receive by being handed a
``SolarForecast`` explicitly. This module supplies *the calendar*, which every
controller may read freely. Conflating the two would let "hours until sunrise"
smuggle in "hours until it stops raining", and the whole causality argument in
:mod:`flexcompute.control` would collapse.

The distinction is enforced by construction: nothing here ever reads measured
irradiance. :func:`solar_elevation_deg` takes a site's coordinates and its time
index and returns solar position. If you find yourself wanting to pass weather
into this module, you are building a forecast and it belongs elsewhere.

Wrapping is legitimate here
---------------------------
:func:`hours_until_next_sunrise` wraps around the end of the year, so 31
December looks forward into 1 January. For a *weather* series that would be
acausality (see :meth:`PerfectSolarForecast.horizon`, which refuses to do it).
For an ephemeris it is simply correct: the sunrise time on 1 January is known
to anyone with a calendar, in December or in any other month.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760

#: Solar elevation above which the sun counts as "up" for rationing purposes.
#: Zero is the geometric horizon. A PV array produces essentially nothing in the
#: first degree of elevation, but the governor only uses this to decide *when to
#: stop rationing*, and being a few minutes early costs nothing — the amount of
#: energy at stake in the first degree is negligible against a night's budget.
DEFAULT_SUNRISE_ELEVATION_DEG = 0.0

#: Solar position is evaluated at the *midpoint* of each hourly interval, to
#: match the convention that hourly irradiance is an interval mean rather than
#: an instantaneous top-of-hour sample.
HOUR_MIDPOINT_OFFSET = pd.Timedelta(minutes=30)


def solar_elevation_deg(
    latitude: float, longitude: float, index: pd.DatetimeIndex
) -> np.ndarray:
    """Solar elevation (degrees above horizon) at the midpoint of each hour.

    Pure ephemeris: no weather, no measurement, no site data beyond position and
    time. ``index`` must be timezone-aware so that "local standard time" means
    something.
    """
    import pvlib

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"index must be a DatetimeIndex, got {type(index).__name__}")
    if index.tz is None:
        raise ValueError(
            "index must be timezone-aware; a naive index makes solar position ambiguous"
        )
    position = pvlib.solarposition.get_solarposition(
        index + HOUR_MIDPOINT_OFFSET, latitude, longitude
    )
    return np.asarray(position["apparent_elevation"].to_numpy(), dtype=float)


def daylight_mask(
    elevation_deg: np.ndarray, threshold_deg: float = DEFAULT_SUNRISE_ELEVATION_DEG
) -> np.ndarray:
    """True where the sun is above ``threshold_deg``."""
    return np.asarray(elevation_deg, dtype=float) >= threshold_deg


def hours_until_next_sunrise(
    elevation_deg: np.ndarray, threshold_deg: float = DEFAULT_SUNRISE_ELEVATION_DEG
) -> np.ndarray:
    """Hours from each timestep to the **next** sunrise, strictly in the future.

    A sunrise is an upward crossing of ``threshold_deg``: hour ``j`` where the
    sun is up and it was down at ``j-1``. The result is therefore always at
    least 1, and during daylight it points at *tomorrow's* sunrise rather than
    at zero.

    That is deliberate and it is what makes a night-rationing rule work in the
    daytime too. A controller asking this question at noon under thick overcast
    is asking "if the sun does not usefully return today, how long must this
    battery last?" — and the honest answer, without a weather forecast, is
    "until tomorrow morning". A rule that returned zero during daylight would
    let the governor spend freely through an overcast day and then meet dusk
    with an empty battery, which is exactly the failure mode the governor
    exists to avoid.

    Wraps at the year boundary; see the module docstring for why that is sound
    for an ephemeris and not for weather.
    """
    up = daylight_mask(elevation_deg, threshold_deg)
    n = len(up)
    if n == 0:
        return np.empty(0, dtype=float)
    if not up.any():
        raise ValueError("the sun never rises in this series; cannot locate a sunrise")

    sunrise = up & ~np.roll(up, 1)
    if not sunrise.any():
        raise ValueError("no upward crossing found; series may be entirely daylight")

    # Backward scan over two concatenated copies so the tail of the year can see
    # January's sunrise. Astronomy is periodic and known in advance.
    doubled = np.concatenate([sunrise, sunrise])
    next_sunrise = np.full(2 * n, -1, dtype=np.int64)
    seen = -1
    for i in range(2 * n - 1, -1, -1):
        if doubled[i]:
            seen = i
        next_sunrise[i] = seen

    out = np.empty(n, dtype=float)
    for t in range(n):
        # strictly future: start looking at t+1
        j = next_sunrise[t + 1]
        if j < 0:
            raise ValueError("failed to find a following sunrise after wrapping")
        out[t] = float(j - t)
    return out


@dataclass(frozen=True)
class SolarClock:
    """A site's ephemeris, precomputed for one year of hourly steps.

    Everything here is knowable in advance. Handing this to a controller grants
    it a calendar, not a forecast.
    """

    elevation_deg: np.ndarray
    hours_to_sunrise: np.ndarray
    latitude: float
    longitude: float
    sunrise_elevation_deg: float = DEFAULT_SUNRISE_ELEVATION_DEG

    def __post_init__(self) -> None:
        if self.elevation_deg.shape != self.hours_to_sunrise.shape:
            raise ValueError("elevation and hours-to-sunrise must be the same length")

    @property
    def is_daylight(self) -> np.ndarray:
        return daylight_mask(self.elevation_deg, self.sunrise_elevation_deg)

    @property
    def daylight_hours(self) -> int:
        return int(self.is_daylight.sum())

    def metadata(self) -> dict:
        return {
            "kind": "ephemeris",
            "latitude": self.latitude,
            "longitude": self.longitude,
            "sunrise_elevation_deg": self.sunrise_elevation_deg,
            "daylight_hours_per_year": self.daylight_hours,
            "mean_hours_to_sunrise": float(self.hours_to_sunrise.mean()),
            "note": (
                "Solar geometry only. Contains no weather information and is "
                "knowable in advance from an ephemeris."
            ),
        }


def build_solar_clock(
    site, *, sunrise_elevation_deg: float = DEFAULT_SUNRISE_ELEVATION_DEG
) -> SolarClock:
    """Build the ephemeris for a :class:`~flexcompute.scenario.Site`.

    Reads only the site's coordinates and its time index — never its irradiance.
    """
    index = site.tmy.data.index
    elevation = solar_elevation_deg(site.tmy.latitude, site.tmy.longitude, index)
    return SolarClock(
        elevation_deg=elevation,
        hours_to_sunrise=hours_until_next_sunrise(elevation, sunrise_elevation_deg),
        latitude=float(site.tmy.latitude),
        longitude=float(site.tmy.longitude),
        sunrise_elevation_deg=float(sunrise_elevation_deg),
    )
