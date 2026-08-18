"""Running the same experiment against many real weather years.

Three jobs, all of which exist because a single typical year cannot answer the
question this project is asking.

**Year sets.** A study is a named, ordered set of actual calendar years from one
source. Mixing sources inside a set is refused rather than warned about: a
change of instrument between two years would show up in the results as weather
variation, and no amount of downstream care recovers from that.

**Droughts.** For an islanded plant the binding event is not a cloudy year, it
is a cloudy *week*. :func:`worst_solar_window` finds the least-sunny contiguous
window of a given length, and :func:`drought_profile` reports several lengths at
once because they can point at different events.

**Aggregation.** Fifteen numbers summarised as mean/median/min/max/P10/P90.
Reported together, always: the headline of this study is as much about spread as
about level, and a mean alone would hide the case where an advantage is real in
two hard years and absent in thirteen easy ones.

A note on what is *not* here: there is deliberately no function that averages
weather before simulation. Averaging removes the tail events that size the
plant, which is the entire reason a TMY was insufficient.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .scenario import Scenario

HOURS_PER_YEAR = 8760

#: The Dallas study window: fifteen consecutive actual years.
DALLAS_STUDY_YEARS: tuple[int, ...] = tuple(range(2010, 2025))

#: Window lengths probed for solar droughts, in hours. One day through one week.
DROUGHT_WINDOWS_H: tuple[int, ...] = (24, 48, 72, 120, 168)


# ---------------------------------------------------------------------------
# Year sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YearSet:
    """A named set of actual weather years drawn from a single source."""

    years: tuple[int, ...]
    source: str = "open_meteo_era5"
    location: str = "dallas"
    label: str = "dallas_2010_2024_era5"

    def __post_init__(self) -> None:
        if len(set(self.years)) != len(self.years):
            raise ValueError("year set contains duplicates")
        if list(self.years) != sorted(self.years):
            raise ValueError("years must be in ascending order")

    def scenarios(self, **scenario_kwargs) -> list[Scenario]:
        return [
            Scenario(
                location=self.location,
                weather_year=year,
                historical_weather_source=self.source,
                **scenario_kwargs,
            )
            for year in self.years
        ]

    def metadata(self) -> dict:
        return {
            "label": self.label,
            "location": self.location,
            "source": self.source,
            "years": list(self.years),
            "n_years": len(self.years),
            "note": (
                "Actual calendar years, simulated independently. Never averaged, "
                "never stitched into a composite, never mixed across sources."
            ),
        }


DALLAS_ERA5 = YearSet(years=DALLAS_STUDY_YEARS)


# ---------------------------------------------------------------------------
# Droughts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolarDrought:
    """The least-sunny contiguous window of a given length in one year."""

    window_hours: int
    start_hour: int
    start_label: str
    mean_solar_fraction: float      # of nameplate DC, on the common scale
    annual_mean_fraction: float

    @property
    def severity(self) -> float:
        """Window output as a fraction of the year's average. Lower is worse."""
        if self.annual_mean_fraction <= 0:
            return 0.0
        return self.mean_solar_fraction / self.annual_mean_fraction

    def as_dict(self) -> dict:
        return {**asdict(self), "severity_vs_annual_mean": self.severity}


def worst_solar_window(site, window_hours: int) -> SolarDrought:
    """Find the least-sunny contiguous ``window_hours`` in a site-year.

    Searched on the **wrapped** series, so a drought straddling 31 December is
    found rather than being cut in half by the calendar. That is not a foresight
    question -- it is post-hoc analysis of a completed year -- and the cyclic SOC
    boundary condition means the simulator treats the year as a loop anyway.
    """
    profile = np.asarray(site.solar_p_dc, dtype=float)
    n = len(profile)
    if not 1 <= window_hours <= n:
        raise ValueError(f"window_hours must lie in 1..{n}, got {window_hours}")

    wrapped = np.concatenate([profile, profile[: window_hours - 1]])
    means = np.convolve(wrapped, np.ones(window_hours) / window_hours, mode="valid")[:n]
    start = int(means.argmin())

    index = site.tmy.data.index
    return SolarDrought(
        window_hours=window_hours,
        start_hour=start,
        start_label=index[start].strftime("%d %b %H:%M"),
        mean_solar_fraction=float(means[start]),
        annual_mean_fraction=float(profile.mean()),
    )


def drought_profile(
    site, windows: Sequence[int] = DROUGHT_WINDOWS_H
) -> dict[int, SolarDrought]:
    """Worst window at several lengths. They need not be the same event."""
    return {int(w): worst_solar_window(site, int(w)) for w in windows}


def year_summary(site) -> dict:
    """Cheap weather-only description of a site-year, for cross-year tables."""
    profile = np.asarray(site.solar_p_dc, dtype=float)
    droughts = drought_profile(site)
    return {
        "weather_year": site.weather_year,
        "weather_source": site.tmy.source,
        "leap_day_dropped": site.tmy.leap_day_dropped,
        "solar_capacity_factor": float(profile.mean()),
        "solar_peak_fraction_of_nameplate": float(profile.max()),
        "annual_pue": float(site.annual_pue),
        "it_load_avg_mw": site.it_load_avg_mw,
        **site.tmy.summary(),
        "droughts": {str(w): d.as_dict() for w, d in droughts.items()},
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(values: Iterable[float]) -> dict:
    """Mean, median, min, max, P10 and P90 of a sample of years.

    All six are reported together on purpose. This study's central question is
    whether a controller advantage is consistent or is carried by two or three
    bad years, and only the spread answers that; a mean on its own is exactly
    the statistic that would hide it.
    """
    array = np.asarray([v for v in values], dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        empty = {k: float("nan") for k in
                 ("mean", "median", "min", "max", "p10", "p90", "std")}
        return {**empty, "n": 0}
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p10": float(np.percentile(finite, 10)),
        "p90": float(np.percentile(finite, 90)),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "n": int(finite.size),
    }


def aggregate_over_years(
    per_year: dict[int, dict], keys: Sequence[str]
) -> dict[str, dict]:
    """Aggregate several metrics across a ``{year: metrics}`` mapping."""
    return {
        key: aggregate(metrics[key] for metrics in per_year.values() if key in metrics)
        for key in keys
    }


def concentration_of_advantage(per_year: dict[int, float]) -> dict:
    """How much of a total advantage comes from the best few years.

    The direct test of "is this consistent, or is it two bad weather years?".
    Reports the share of the summed *positive* advantage contributed by the top
    1, 2 and 3 years, and how many years show any advantage at all.

    Summing only the positive contributions matters. A strategy that gains in
    three years and loses in twelve has a small or negative total, and dividing
    by that total produces shares above 100% or with a flipped sign — a number
    that looks like an answer and is not one. The shares are therefore defined
    against the gross gain, and are ``nan`` when there is no gain to apportion.

    A perfectly even spread over 15 years puts 20% in the top 3;
    ``even_split_top_3_share`` gives the comparison point for the actual sample
    size, so a small exploratory run does not read as concentrated.
    """
    values = np.asarray(list(per_year.values()), dtype=float)
    finite = values[np.isfinite(values)]
    n = int(finite.size)

    gains = finite[finite > 0]
    gross_gain = float(gains.sum())
    order = np.sort(gains)[::-1]

    shares: dict[str, float] = {}
    for k in (1, 2, 3):
        shares[f"top_{k}_year_share"] = (
            float(order[: min(k, len(order))].sum() / gross_gain)
            if gross_gain > 0 else float("nan")
        )
    return {
        **shares,
        "gross_gain": gross_gain,
        "net_advantage": float(finite.sum()),
        "years_with_positive_advantage": int(gains.size),
        "years_evaluated": n,
        "even_split_top_3_share": (min(3, n) / n) if n else float("nan"),
        "interpretable": bool(gross_gain > 0 and n >= 3),
    }


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------
#
# Runs are embarrassingly parallel across years and sizings, and the expensive
# ones (receding-horizon MPC) are single-threaded, so a process pool is close to
# linear in cores. Workers rebuild the Site from its Scenario rather than
# receiving one: a Site holds live upstream objects, and pickling those across a
# process boundary is both fragile and slower than the rebuild (~0.7 s).

def default_workers() -> int:
    """Leave a couple of cores for the machine to stay usable."""
    return max(1, (os.cpu_count() or 2) - 2)


@dataclass
class Job:
    """One (scenario, sizing, strategy) evaluation."""

    scenario: Scenario
    strategy: str
    solar_mw: float
    battery_mw: float
    duration_h: float
    run_kwargs: dict = field(default_factory=dict)
    tag: tuple = ()

    def key(self) -> tuple:
        return (self.scenario.weather_year, self.strategy) + self.tag


def _execute(job: Job) -> tuple[tuple, dict]:
    """Worker entry point. Must stay importable at module level for pickling."""
    import logging

    from .experiments import Sizing, run_strategy

    # Each worker is a fresh interpreter, so the parent's logging suppression
    # does not carry over and upstream's INFO chatter would drown the progress
    # output at 15 years x 5 scales x 6 strategies.
    logging.disable(logging.INFO)

    site = job.scenario.build()
    sizing = Sizing(job.solar_mw, job.battery_mw, job.duration_h)
    result = run_strategy(site, job.strategy, sizing, **job.run_kwargs)
    metrics = dict(result.metrics)
    metrics["_weather_year"] = job.scenario.weather_year
    metrics["_weather_source"] = site.tmy.source
    metrics["_strategy"] = job.strategy
    metrics["_sizing"] = sizing.as_dict()
    metrics["_curve"] = result.metadata["gpu"]["curve"]["name"]
    metrics["_curve_kind"] = result.metadata["gpu"]["curve"]["kind"]
    metrics["_cooling_fixed_fraction"] = result.metadata["cooling_fixed_fraction"]
    return job.key(), metrics


def run_jobs(
    jobs: Sequence[Job],
    *,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int, tuple], None]] = None,
) -> dict[tuple, dict]:
    """Execute jobs in parallel, returning ``{job.key(): metrics}``.

    Falls back to serial execution when ``workers == 1``, which keeps
    debugging and profiling straightforward.
    """
    workers = default_workers() if workers is None else workers
    results: dict[tuple, dict] = {}

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            key, metrics = _execute(job)
            results[key] = metrics
            if progress:
                progress(i, len(jobs), key)
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_execute, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            key, metrics = future.result()
            results[key] = metrics
            if progress:
                progress(i, len(jobs), key)
    return results
