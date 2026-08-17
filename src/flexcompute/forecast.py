"""Solar forecasts: what a controller is allowed to believe about the future.

This is deliberately a separate module from :mod:`flexcompute.weather`. Weather
supplies **ground truth** -- what actually happens, and what the dispatcher
simulates. A forecast supplies **a belief** -- what the controller gets to see
when it plans. Keeping them apart is the mechanism that stops a forecast-aware
experiment from quietly becoming a perfect-foresight one.

The rule the architecture enforces: :class:`~flexcompute.control.Observation`
carries no future information at all, so a controller can only anticipate if it
was *handed* a forecast object at construction. Perfect foresight is therefore
an explicit dependency you can see in the call site, never an accident.

Two implementations:

:class:`PerfectSolarForecast`
    The truth. The theoretical upper bound, not an achievable design. Every
    result computed with it must say so.
:class:`NoisySolarForecast`
    A believable day-ahead forecast: seeded, temporally correlated, with error
    that grows with lead time and vanishes at lead zero. The gap between the two
    is the question Milestone 6 exists to answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

HOURS_PER_DAY = 24


@runtime_checkable
class SolarForecast(Protocol):
    """Expected PV production over a horizon starting at ``t``."""

    name: str

    def horizon(self, t: int, hours: int) -> np.ndarray:
        """Return expected DC MW for hours ``t .. t+hours-1``.

        Implementations must be deterministic given their seed, and must not
        depend on anything the controller could not know at time ``t`` unless
        they are explicitly a perfect-foresight implementation.
        """
        ...

    def metadata(self) -> dict: ...


@dataclass
class PerfectSolarForecast:
    """The future, exactly. An upper bound, not a design.

    Wraps the realised solar series and hands back the truth. Any controller
    using this is measuring *the value of information* -- the ceiling a real
    forecast could approach but never reach.
    """

    actual_dc_mw: np.ndarray
    name: str = "perfect"

    def __post_init__(self) -> None:
        self.actual_dc_mw = np.asarray(self.actual_dc_mw, dtype=float)
        if self.actual_dc_mw.ndim != 1:
            raise ValueError("actual_dc_mw must be one-dimensional")

    def horizon(self, t: int, hours: int) -> np.ndarray:
        """Truth for ``t .. t+hours-1``, **truncated** at the end of the series.

        Never wrapped: December must not see January's sunshine, which would be
        acausality rather than foresight. Never zero-padded either -- that
        invents a week-long night at year end and makes the planner panic. A
        short window at the boundary is the honest answer.
        """
        return self.actual_dc_mw[t : t + hours]

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "kind": "perfect_foresight",
            "caveat": (
                "Exact future knowledge. Establishes the theoretical ceiling; "
                "not achievable by any real forecasting system."
            ),
        }


# ---------------------------------------------------------------------------
# A forecast that can be wrong
# ---------------------------------------------------------------------------

def clear_sky_envelope(actual_dc_mw: np.ndarray, window_days: int = 15) -> np.ndarray:
    """An empirical clear-sky ceiling: the best this hour-of-day gets, locally.

    For each hour of each day, the maximum production observed at the *same
    hour of day* within ``±window_days``. That is a robust proxy for the
    clear-sky curve: it carries the diurnal and seasonal shape, it is exactly
    zero at night, and it needs no irradiance model or extra dependency.

    Why the error model needs this at all: an error expressed as a fraction of
    *realised* production is silently perfect during a multi-day drought --
    zero production means zero error, so the forecast would nail precisely the
    events that size an islanded plant. Scaling error by the clear-sky
    potential instead lets the forecast say "sunny" on a day that turns out
    overcast, which is the failure mode that actually costs a controller.

    The envelope is **climatology, not weather**: it says what a clear day in
    mid-March looks like at this site, which any operator knows from a
    datasheet and a calendar. Only the deviation from it is treated as unknown.
    """
    actual = np.asarray(actual_dc_mw, dtype=float)
    if window_days < 1:
        raise ValueError("window_days must be at least 1")
    n = len(actual)
    pad = (-n) % HOURS_PER_DAY
    grid = np.concatenate([actual, np.zeros(pad)]).reshape(-1, HOURS_PER_DAY)

    envelope = np.empty_like(grid)
    for day in range(len(grid)):
        lo = max(0, day - window_days)
        hi = min(len(grid), day + window_days + 1)
        envelope[day] = grid[lo:hi].max(axis=0)
    return envelope.reshape(-1)[:n]


def _correlated_noise(n: int, correlation_hours: float, seed: int) -> np.ndarray:
    """Unit-variance AR(1) noise with an exponential autocorrelation.

    White noise would be the wrong shape and, worse, the *easy* shape: an MPC
    integrating over a 48-hour horizon barely notices zero-mean independent
    hourly errors, because they cancel inside the plan. Real forecast errors do
    not cancel -- when a front is mistimed the whole day is wrong in the same
    direction -- so the error has to persist on the timescale of the weather.
    """
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal(n)
    if correlation_hours <= 0:
        return draws
    phi = float(np.exp(-1.0 / correlation_hours))
    innovation = float(np.sqrt(1.0 - phi * phi))
    z = np.empty(n)
    z[0] = draws[0]
    for i in range(1, n):
        z[i] = phi * z[i - 1] + innovation * draws[i]
    return z


@dataclass
class NoisySolarForecast:
    """A forecast that is wrong in the ways real forecasts are wrong.

    The belief about hour ``t + k``, issued at ``t``, is

        ``truth[t+k] + envelope[t+k] * w(k) * (sigma_24h * z[t+k] + bias)``

    clipped to ``[0, envelope]``. Three structural choices, each of which
    changes what a controller can do about it:

    **Error scales with the clear-sky envelope, not with realised output.**
    So a drought can be missed (see :func:`clear_sky_envelope`), and nights
    carry no error at all -- nobody mis-forecasts darkness.

    **Error grows with lead time and is exactly zero at lead zero.**
    ``w(k) = (min(k, saturation_hours) / 24) ** lead_exponent``, so ``sigma_24h``
    is by definition the error at one day ahead, the square-root default is the
    usual diffusive growth, and beyond ``saturation_hours`` the forecast has
    decayed to climatology and stops getting worse. Lead zero being exact is
    not a convenience: the controller measures current irradiance, and
    :class:`~flexcompute.control.Observation` already gives it that.

    **Error is temporally correlated** on the timescale of weather
    (``correlation_hours``), so a badly-forecast day is wrong all day.

    One deliberate simplification, and it is optimistic: the error for a given
    target hour keeps its sign and shape and merely *shrinks* as that hour
    approaches, so successive forecasts converge monotonically on the truth.
    Real forecast revisions jitter. Pulling in the other direction, the error
    is fully persistent across issue times -- re-planning cannot average it
    away, which is the pessimistic half of the same choice. See ASSUMPTIONS B13.

    ``sigma_24h`` is a fraction of the *hour's clear-sky potential*, not of
    plant capacity, so it does not compare directly with published nRMSE
    figures. :meth:`error_stats` reports the realised whole-year error in the
    conventional normalisations, and that is what results should quote.
    """

    actual_dc_mw: np.ndarray
    sigma_24h: float = 0.15
    lead_exponent: float = 0.5
    saturation_hours: float = 72.0
    correlation_hours: float = 24.0
    bias: float = 0.0
    envelope_window_days: int = 15
    seed: int = 20260815
    envelope_dc_mw: Optional[np.ndarray] = None
    name: str = "noisy"

    _envelope: np.ndarray = field(init=False, repr=False)
    _z: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.actual_dc_mw = np.asarray(self.actual_dc_mw, dtype=float)
        if self.actual_dc_mw.ndim != 1:
            raise ValueError("actual_dc_mw must be one-dimensional")
        if self.sigma_24h < 0:
            raise ValueError("sigma_24h must be non-negative")
        if self.saturation_hours <= 0:
            raise ValueError("saturation_hours must be positive")

        if self.envelope_dc_mw is None:
            self._envelope = clear_sky_envelope(self.actual_dc_mw, self.envelope_window_days)
        else:
            self._envelope = np.asarray(self.envelope_dc_mw, dtype=float)
            if self._envelope.shape != self.actual_dc_mw.shape:
                raise ValueError("envelope_dc_mw must have the same shape as actual_dc_mw")
            if np.any(self._envelope < self.actual_dc_mw - 1e-9):
                raise ValueError("envelope_dc_mw must be an upper bound on actual_dc_mw")
        self._z = _correlated_noise(len(self.actual_dc_mw), self.correlation_hours, self.seed)

    # -- the error model -------------------------------------------------

    def _lead_weight(self, lead_hours: np.ndarray) -> np.ndarray:
        capped = np.minimum(np.asarray(lead_hours, dtype=float), self.saturation_hours)
        return (capped / float(HOURS_PER_DAY)) ** self.lead_exponent

    def _believe(self, target: slice, weight: np.ndarray) -> np.ndarray:
        envelope = self._envelope[target]
        deviation = envelope * (self.sigma_24h * weight * self._z[target] + self.bias * weight)
        return np.clip(self.actual_dc_mw[target] + deviation, 0.0, envelope)

    def horizon(self, t: int, hours: int) -> np.ndarray:
        """Belief about hours ``t .. t+hours-1``, held at ``t``.

        Truncated at the end of the series for the same reasons as
        :meth:`PerfectSolarForecast.horizon`: padding invents a week-long
        December night, wrapping is acausal.

        A pure function of ``(t, hours)`` and the seed -- asking twice gives the
        same answer, which is what makes a run reproducible.
        """
        n = len(self.actual_dc_mw)
        if t >= n or hours <= 0:
            return np.empty(0, dtype=float)
        end = min(t + hours, n)
        return self._believe(slice(t, end), self._lead_weight(np.arange(end - t)))

    def forecast_at_lead(self, lead_hours: int) -> np.ndarray:
        """The whole year as believed ``lead_hours`` in advance.

        Not used by the controller -- it is the diagnostic view, for scoring the
        forecast the way a forecaster would rather than the way a planner uses it.
        """
        weight = self._lead_weight(np.full(len(self.actual_dc_mw), float(lead_hours)))
        return self._believe(slice(None), weight)

    def error_stats(self, lead_hours: tuple[int, ...] = (1, 6, 12, 24, 48)) -> dict:
        """Realised forecast error, in the normalisations forecasters publish.

        Scored over **daylight hours only** (``envelope > 0``). Including the
        night would divide the same error by three times as many hours and
        halve every number; excluding it is the stricter and more comparable
        read.
        """
        daylight = self._envelope > 0.0
        capacity = float(self._envelope.max()) if len(self._envelope) else 0.0
        mean_output = float(self.actual_dc_mw[daylight].mean()) if daylight.any() else 0.0

        stats: dict = {
            "capacity_mw_dc": capacity,
            "mean_daylight_output_mw_dc": mean_output,
            "daylight_hours": int(daylight.sum()),
        }
        for k in lead_hours:
            error = (self.forecast_at_lead(k) - self.actual_dc_mw)[daylight]
            rmse = float(np.sqrt(np.mean(error**2))) if error.size else 0.0
            mae = float(np.mean(np.abs(error))) if error.size else 0.0
            stats[f"lead_{k}h"] = {
                "rmse_mw": rmse,
                "mae_mw": mae,
                "bias_mw": float(np.mean(error)) if error.size else 0.0,
                "nrmse_pct_of_capacity": 100.0 * rmse / capacity if capacity else 0.0,
                "nmae_pct_of_capacity": 100.0 * mae / capacity if capacity else 0.0,
                "nrmse_pct_of_mean_output": 100.0 * rmse / mean_output if mean_output else 0.0,
            }
        return stats

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "kind": "noisy",
            "parameters": {
                "sigma_24h": self.sigma_24h,
                "lead_exponent": self.lead_exponent,
                "saturation_hours": self.saturation_hours,
                "correlation_hours": self.correlation_hours,
                "bias": self.bias,
                "envelope_window_days": self.envelope_window_days,
                "seed": self.seed,
            },
            "realised_error": self.error_stats(),
            "caveat": (
                "Synthetic error model, not a validated forecasting system. "
                "sigma_24h is a fraction of each hour's clear-sky potential; "
                "quote realised_error for comparison with published skill. "
                "Successive forecasts of the same hour shrink monotonically "
                "toward the truth (optimistic) and never average away across "
                "re-plans (pessimistic). See ASSUMPTIONS B13."
            ),
        }
