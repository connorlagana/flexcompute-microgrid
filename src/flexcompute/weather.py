"""Weather (TMY) providers behind a single interface.

Why this exists
---------------
The reference model fetches NSRDB PSM4 TMY data through the NLR API, which
needs a free API key. That is the right source to match the published paper,
but it makes the project unrunnable for anyone without credentials and blocks
CI. So weather acquisition is put behind an interface with two backends:

    NSRDBProvider  -- upstream's own loader (needs NLR_API_KEY/NLR_EMAIL,
                      or a warm cache). Matches the paper.
    PVGISProvider  -- pvlib's PVGIS TMY endpoint. No key, no account.

Both return the same schema, so every downstream stage is identical. The
provider name is stamped into every result and snapshot, because *weather
source is an experimental variable, not an implementation detail*: absolute MW
and MWh figures shift between sources. Controller-vs-controller comparisons
are only valid within a single source.

This is deliberately separate from the (future) forecast interface. This module
supplies **ground truth** -- what actually happens. A ``SolarForecast`` supplies
**what a controller is allowed to believe**. Conflating the two is how a
forecast-aware experiment accidentally becomes a perfect-foresight one.

Time convention
---------------
Every provider returns 8760 rows indexed by *local standard time* (no daylight
saving), starting at 00:00 on 1 January of a synthetic non-leap reference year.
Row ``i`` is therefore local hour-of-year ``i``, which is the convention the
downstream positional arrays (load shape, PUE profile, solar profile) already
assume, and the convention controllers need in order to reason about "hours
until sunrise".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from .upstream_bridge import PROJECT_ROOT

# Columns every downstream consumer (pvlib ModelChain, the PUE lookup) needs.
REQUIRED_COLUMNS = ("temp_air", "relative_humidity", "ghi", "dni", "dhi", "wind_speed")

#: Synthetic non-leap year used for the canonical local-time index.
REFERENCE_YEAR = 2023

HOURS_PER_YEAR = 8760

CACHE_DIR = PROJECT_ROOT / "data" / "weather"


@dataclass(frozen=True)
class WeatherYear:
    """One year of site weather plus its provenance.

    Covers both kinds of year, and the distinction is carried in ``year``:

    ``year is None``
        A **typical meteorological year** — a synthetic composite assembled
        from months drawn from different real years. Smooth, representative,
        and systematically short of the multi-day extremes that size an
        islanded plant, because a TMY selection algorithm picks *typical*
        months by construction.
    ``year is an int``
        An **actual historical year**. What really happened, droughts included.

    Conflating the two is the specific error this field exists to prevent: a
    controller advantage measured on a TMY is measured on weather with the hard
    parts averaged out.
    """

    data: pd.DataFrame
    source: str
    latitude: float
    longitude: float
    timezone_name: str
    utc_offset_hours: float
    retrieved_utc: str
    notes: str = ""
    year: Optional[int] = None
    dataset: str = ""
    leap_day_dropped: bool = False

    @property
    def is_typical(self) -> bool:
        """True for a synthetic TMY, False for an actual historical year."""
        return self.year is None

    def provenance(self) -> dict:
        """Provenance dict for snapshots (everything except the bulk data)."""
        return {
            "source": self.source,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone_name": self.timezone_name,
            "utc_offset_hours": self.utc_offset_hours,
            "retrieved_utc": self.retrieved_utc,
            "notes": self.notes,
            "year": self.year,
            "dataset": self.dataset,
            "leap_day_dropped": self.leap_day_dropped,
        }

    def summary(self) -> dict:
        """Cheap scalar summary, useful as a regression fingerprint."""
        d = self.data
        return {
            "rows": int(len(d)),
            "ghi_annual_kwh_per_m2": float(d["ghi"].sum() / 1000.0),
            "dni_annual_kwh_per_m2": float(d["dni"].sum() / 1000.0),
            "temp_air_mean_c": float(d["temp_air"].mean()),
            "temp_air_max_c": float(d["temp_air"].max()),
            "relative_humidity_mean_pct": float(d["relative_humidity"].mean()),
        }


#: Backwards-compatible alias. Older code and cached snapshots call this a TMY;
#: it is the same object, and ``year`` distinguishes the two cases.
TMY = WeatherYear


class WeatherProvider(Protocol):
    """Anything that can supply a typical year for a site."""

    name: str

    def fetch(self, latitude: float, longitude: float) -> WeatherYear: ...


class HistoricalWeatherProvider(Protocol):
    """Anything that can supply one *actual* calendar year for a site."""

    name: str

    def available_years(self) -> tuple[int, int]:
        """Inclusive ``(first, last)`` calendar year this source covers."""
        ...

    def fetch_year(self, latitude: float, longitude: float, year: int) -> WeatherYear: ...


# ---------------------------------------------------------------------------
# Local-time normalisation
# ---------------------------------------------------------------------------

def _standard_utc_offset_hours(latitude: float, longitude: float) -> tuple[str, float]:
    """Return (IANA tz name, *standard* UTC offset in hours) for a site.

    Standard time, not daylight saving: TMY files are conventionally in local
    standard time year-round, and a DST discontinuity would corrupt the
    hour-of-year indexing that every positional array here relies on. The
    offset is sampled on 1 January, which is standard time in both hemispheres'
    usual DST regimes for the US sites in scope.
    """
    from zoneinfo import ZoneInfo

    import tzfpy

    tz_name = tzfpy.get_tz(longitude, latitude) or "UTC"
    january = datetime(REFERENCE_YEAR, 1, 1, 12, tzinfo=ZoneInfo(tz_name))
    offset = january.utcoffset()
    return tz_name, (offset.total_seconds() / 3600.0 if offset else 0.0)


def to_local_standard_hour_index(
    df: pd.DataFrame, utc_offset_hours: float
) -> pd.DataFrame:
    """Reindex a TMY frame onto local-standard hour-of-year 0..8759.

    Handles the fact that different providers time-stamp TMY data differently
    (PVGIS returns UTC; NSRDB returns local standard time) and that TMY months
    are stitched together from different calendar years. Rows are placed by
    their local (month, day, hour), so the result is always ordered
    1 Jan 00:00 -> 31 Dec 23:00 in local standard time.

    A 29 February row (possible when a TMY month is drawn from a leap year) is
    dropped, since the canonical index is a non-leap year.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"TMY index must be a DatetimeIndex, got {type(idx).__name__}")

    tz = timezone(timedelta(hours=utc_offset_hours))
    if idx.tz is None:
        # No tz information: assume the provider already meant local standard time.
        local = idx
    else:
        local = idx.tz_convert(tz)

    keep = ~((local.month == 2) & (local.day == 29))
    frame = df.loc[keep].copy()
    local = local[keep]

    hour_of_year = (
        pd.to_datetime(
            {
                "year": np.full(len(local), REFERENCE_YEAR),
                "month": local.month,
                "day": local.day,
                "hour": local.hour,
            }
        )
        - pd.Timestamp(f"{REFERENCE_YEAR}-01-01")
    ) // pd.Timedelta(hours=1)

    frame = frame.assign(_hoy=np.asarray(hour_of_year)).sort_values("_hoy")
    if frame["_hoy"].duplicated().any():
        raise ValueError("Duplicate local hour-of-year rows in TMY data")
    frame = frame.drop(columns="_hoy")

    if len(frame) != HOURS_PER_YEAR:
        raise ValueError(f"Expected {HOURS_PER_YEAR} TMY rows, got {len(frame)}")

    frame.index = pd.date_range(
        start=f"{REFERENCE_YEAR}-01-01 00:00", periods=HOURS_PER_YEAR, freq="h", tz=tz
    )
    frame.index.name = "local_standard_time"
    return frame


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"TMY data missing required columns: {missing}")
    out = df[list(REQUIRED_COLUMNS)].astype(float)
    if out.isna().any().any():
        raise ValueError("TMY data contains NaNs")
    return out


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@dataclass
class PVGISProvider:
    """TMY from PVGIS via pvlib. No API key required.

    PVGIS serves NSRDB-derived satellite data for the Americas, but its TMY is
    assembled independently of NREL/NLR's PSM4 TMY, so annual irradiance
    differs by a few percent. Fine for controller-vs-controller comparison;
    not interchangeable with the published paper's absolute numbers.
    """

    name: str = "pvgis"

    def fetch(self, latitude: float, longitude: float) -> WeatherYear:
        import pvlib

        tz_name, offset = _standard_utc_offset_hours(latitude, longitude)
        raw = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)[0]
        frame = to_local_standard_hour_index(_validate(raw), offset)
        return WeatherYear(
            data=frame,
            source=self.name,
            latitude=latitude,
            longitude=longitude,
            timezone_name=tz_name,
            utc_offset_hours=offset,
            retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            notes="PVGIS TMY via pvlib.iotools.get_pvgis_tmy; source timestamps UTC.",
        )


@dataclass
class NSRDBProvider:
    """TMY from NSRDB PSM4 through upstream's own loader.

    Matches the reference paper's weather source. Requires NLR_API_KEY and
    NLR_EMAIL in the environment, or a previously warmed upstream cache under
    ``upstream/output_tables/nsrdb_cache/``.
    """

    name: str = "nsrdb"

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("NLR_API_KEY") and os.environ.get("NLR_EMAIL"))

    def fetch(self, latitude: float, longitude: float) -> WeatherYear:
        from .upstream_bridge import ensure_upstream_importable

        ensure_upstream_importable()
        from nsrdb_loader import get_nsrdb_tmy  # type: ignore[import-not-found]

        tz_name, offset = _standard_utc_offset_hours(latitude, longitude)
        raw = get_nsrdb_tmy(latitude, longitude)
        frame = to_local_standard_hour_index(_validate(raw), offset)
        return WeatherYear(
            data=frame,
            source=self.name,
            latitude=latitude,
            longitude=longitude,
            timezone_name=tz_name,
            utc_offset_hours=offset,
            retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            notes="NSRDB PSM4 TMY via upstream nsrdb_loader.get_nsrdb_tmy.",
        )


PROVIDERS: dict[str, WeatherProvider] = {
    "pvgis": PVGISProvider(),
    "nsrdb": NSRDBProvider(),
}


# ---------------------------------------------------------------------------
# Caching front door
# ---------------------------------------------------------------------------

def _cache_paths(source: str, latitude: float, longitude: float) -> tuple[Path, Path]:
    stem = f"{source}_tmy_{latitude:.4f}_{longitude:.4f}"
    return CACHE_DIR / f"{stem}.parquet", CACHE_DIR / f"{stem}.json"


def get_tmy(
    latitude: float,
    longitude: float,
    source: str = "pvgis",
    *,
    use_cache: bool = True,
    refresh: bool = False,
) -> TMY:
    """Fetch (or load from cache) a normalised TMY for a site.

    Caching is on by default so that every run after the first is fully offline
    and byte-identical -- a precondition for the reproducible baseline.
    """
    if source not in PROVIDERS:
        raise ValueError(f"Unknown weather source '{source}'. Options: {sorted(PROVIDERS)}")

    data_path, meta_path = _cache_paths(source, latitude, longitude)
    if use_cache and not refresh and data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return TMY(data=pd.read_parquet(data_path), **meta)

    tmy = PROVIDERS[source].fetch(latitude, longitude)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmy.data.to_parquet(data_path)
        meta_path.write_text(json.dumps(tmy.provenance(), indent=2) + "\n")
    return tmy


# ---------------------------------------------------------------------------
# Actual historical years
# ---------------------------------------------------------------------------
#
# A TMY under-represents exactly the events that size an islanded plant. Its
# construction picks the most *typical* January from a 20-year record, then the
# most typical February, and so on -- an algorithm that is guaranteed to leave
# out the worst multi-day solar drought in the record, because that drought is
# what made its month atypical. For a system whose battery must ride through
# such an event, that is not a small distortion.
#
# So these providers return what actually happened, one calendar year at a
# time, never averaged and never stitched.

#: Extra hours fetched either side of the target year so that the local-standard
#: -time year can be cut out of a UTC series without missing its ends.
_UTC_MARGIN_DAYS = 2


def _select_local_year(df: pd.DataFrame, utc_offset_hours: float, year: int) -> pd.DataFrame:
    """Cut the rows belonging to local-standard-time calendar ``year``."""
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("historical weather must be indexed by timestamp")
    if idx.tz is None:
        raise ValueError("expected a timezone-aware index before local-year selection")
    local = idx.tz_convert(timezone(timedelta(hours=utc_offset_hours)))
    out = df.loc[np.asarray(local.year == year)]
    out.index = local[local.year == year]
    return out


def _finalise_year(
    frame: pd.DataFrame,
    *,
    utc_offset_hours: float,
    year: int,
) -> tuple[pd.DataFrame, bool]:
    """Validate, drop 29 February if present, and put on the canonical index.

    **Leap-year policy.** The canonical index is 8760 rows of a non-leap
    reference year, because every positional array downstream (the hourly IT
    load shape, the PUE profile, the solar profile) is 8760 long. A leap year is
    reconciled by dropping 29 February entirely -- 24 rows -- rather than by
    truncating the tail.

    Dropping the leap day preserves seasonal alignment: 1 July stays 1 July in
    every year, so a cross-year comparison lines up. Truncating the tail would
    shift every date after February by one day in leap years only, which would
    show up as spurious year-to-year variation in exactly the winter hours the
    study cares about. The cost is 24 hours of late-February weather in 4 of the
    15 years, and the fact is recorded on the result rather than assumed.
    """
    validated = _validate(frame)
    is_leap = bool(((validated.index.month == 2) & (validated.index.day == 29)).any())
    normalised = to_local_standard_hour_index(validated, utc_offset_hours)
    if len(normalised) != HOURS_PER_YEAR:
        raise ValueError(
            f"Historical year {year} yielded {len(normalised)} rows, expected "
            f"{HOURS_PER_YEAR}"
        )
    return normalised, is_leap


@dataclass
class NSRDBHistoricalProvider:
    """Actual years from NSRDB PSM v4 (GOES CONUS), via the NLR download API.

    The satellite-derived source the reference paper uses, and the best
    available irradiance record for the continental US: 4 km, cloud-resolving,
    validated against surface stations.

    Coverage is the constraint. The v4 CONUS product is GOES-era only, so it
    starts in 2018 -- the aggregated v4 endpoint that would reach back to 1998
    returns a server-side processing failure at the time of writing. Fifteen
    years is therefore not obtainable from this source, which is why
    :class:`OpenMeteoHistoricalProvider` exists alongside it.

    Requires ``NLR_API_KEY`` and ``NLR_EMAIL``. ``DEMO_KEY`` works but is rate
    limited to a handful of requests per hour, so it is usable for a spot check
    and not for a bulk download.
    """

    api_key: str = field(default_factory=lambda: os.environ.get("NLR_API_KEY", "DEMO_KEY"))
    email: str = field(default_factory=lambda: os.environ.get("NLR_EMAIL", ""))
    name: str = "nsrdb_psm4_conus"

    FIRST_YEAR: int = 2018
    LAST_YEAR: int = 2024

    ENDPOINT: str = (
        "https://developer.nlr.gov/api/nsrdb/v2/solar/"
        "nsrdb-GOES-conus-v4-0-0-download.csv"
    )

    def available_years(self) -> tuple[int, int]:
        return (self.FIRST_YEAR, self.LAST_YEAR)

    @staticmethod
    def has_credentials() -> bool:
        return bool(os.environ.get("NLR_API_KEY") and os.environ.get("NLR_EMAIL"))

    def fetch_year(self, latitude: float, longitude: float, year: int) -> WeatherYear:
        import requests

        if not self.FIRST_YEAR <= year <= self.LAST_YEAR:
            raise ValueError(
                f"NSRDB PSM4 CONUS covers {self.FIRST_YEAR}-{self.LAST_YEAR}; "
                f"asked for {year}"
            )
        if not self.email:
            raise RuntimeError(
                "NSRDB requires an email address. Set NLR_EMAIL (and ideally "
                "NLR_API_KEY; request a free key at https://developer.nlr.gov/signup/)."
            )

        tz_name, offset = _standard_utc_offset_hours(latitude, longitude)
        response = requests.get(
            self.ENDPOINT,
            params={
                "api_key": self.api_key,
                "wkt": f"POINT({longitude:.4f} {latitude:.4f})",
                "names": str(year),
                "interval": "60",
                # Local standard time at the grid cell, matching this project's
                # convention. NSRDB never applies daylight saving.
                "utc": "false",
                "email": self.email,
                "attributes": "ghi,dni,dhi,air_temperature,relative_humidity,wind_speed",
            },
            timeout=300,
        )
        if response.status_code == 429:
            raise RuntimeError(
                "NSRDB rate limit hit. DEMO_KEY allows only a few requests per "
                "hour; set NLR_API_KEY to a personal key for bulk downloads."
            )
        response.raise_for_status()
        return self.parse(response.text, latitude, longitude, year)

    def parse(
        self, payload: str, latitude: float, longitude: float, year: int
    ) -> WeatherYear:
        """Turn a PSM4 CSV body into a normalised year.

        Split out from the HTTP call so the parsing can be exercised offline
        against a saved response, and so a manually downloaded file can prime
        the cache when the API is rate limited.
        """
        import io

        tz_name, offset = _standard_utc_offset_hours(latitude, longitude)

        # Two metadata rows precede the data header.
        meta = pd.read_csv(io.StringIO(payload), nrows=1)
        raw = pd.read_csv(io.StringIO(payload), skiprows=2)
        raw = raw.rename(
            columns={
                "GHI": "ghi",
                "DNI": "dni",
                "DHI": "dhi",
                "Temperature": "temp_air",
                "Relative Humidity": "relative_humidity",
                "Wind Speed": "wind_speed",
            }
        )
        # Minute is 30 for hourly data (interval midpoint); the row is the mean
        # over the hour beginning at Hour:00, so the hour label is what we want.
        index = pd.to_datetime(raw[["Year", "Month", "Day", "Hour"]])
        raw.index = index.dt.tz_localize(timezone(timedelta(hours=offset)))

        frame, leap = _finalise_year(raw, utc_offset_hours=offset, year=year)
        return WeatherYear(
            data=frame,
            source=self.name,
            latitude=latitude,
            longitude=longitude,
            timezone_name=tz_name,
            utc_offset_hours=offset,
            retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            year=year,
            dataset=f"NSRDB PSM v{meta['Version'].iloc[0]} GOES CONUS, 4 km, hourly",
            leap_day_dropped=leap,
            notes=(
                "Satellite-derived (GOES) cloud-resolving irradiance. Requested "
                "in local standard time; no daylight saving applied."
            ),
        )


@dataclass
class OpenMeteoHistoricalProvider:
    """Actual years from the ERA5 reanalysis, via the Open-Meteo archive API.

    No key, no account, and it covers 1940 to the present, which is what makes a
    15-year study possible at all here.

    The tradeoff is real and runs **against** this project's hypothesis, which
    is the main reason it is acceptable as the primary multi-year source.
    ERA5 is a reanalysis on a ~25 km grid, not a satellite cloud retrieval, so
    it smooths sub-grid cloud structure and is known to over-predict irradiance
    under broken cloud. A smoothed drought is a shallower drought, and shallower
    droughts are precisely the conditions under which flexible control is worth
    *less*. Any controller advantage measured on this source is therefore
    likely to be an under-estimate rather than an artefact.

    :class:`NSRDBHistoricalProvider` is run over the overlapping years as a
    cross-check on exactly that claim.
    """

    name: str = "open_meteo_era5"
    endpoint: str = "https://archive-api.open-meteo.com/v1/archive"

    FIRST_YEAR: int = 1950
    LAST_YEAR: int = 2025

    VARIABLES: tuple[str, ...] = (
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
        "direct_normal_irradiance",
        "diffuse_radiation",
    )

    def available_years(self) -> tuple[int, int]:
        return (self.FIRST_YEAR, self.LAST_YEAR)

    def fetch_year(self, latitude: float, longitude: float, year: int) -> WeatherYear:
        import requests

        tz_name, offset = _standard_utc_offset_hours(latitude, longitude)

        # Fetch in UTC with a margin, then cut the local-standard-time year out
        # of it. Asking the API for a named timezone would apply daylight
        # saving and put a one-hour discontinuity in the middle of the year.
        start = (datetime(year, 1, 1) - timedelta(days=_UTC_MARGIN_DAYS)).date()
        end = (datetime(year, 12, 31) + timedelta(days=_UTC_MARGIN_DAYS)).date()
        response = requests.get(
            self.endpoint,
            params={
                "latitude": f"{latitude:.4f}",
                "longitude": f"{longitude:.4f}",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": ",".join(self.VARIABLES),
                "wind_speed_unit": "ms",
                "timezone": "UTC",
                # Pin the model so a re-fetch is reproducible rather than
                # silently upgraded to whatever "best match" means next year.
                "models": "era5",
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()

        hourly = payload["hourly"]
        raw = pd.DataFrame(
            {
                "temp_air": hourly["temperature_2m"],
                "relative_humidity": hourly["relative_humidity_2m"],
                "wind_speed": hourly["wind_speed_10m"],
                "ghi": hourly["shortwave_radiation"],
                "dni": hourly["direct_normal_irradiance"],
                "dhi": hourly["diffuse_radiation"],
            },
            index=pd.to_datetime(hourly["time"]).tz_localize("UTC"),
        )
        # ERA5 accumulates radiation over the hour *ending* at the timestamp in
        # some products; Open-Meteo documents its archive hourly values as the
        # mean over the hour *beginning* at the timestamp, which matches the
        # convention used by NSRDB and by this project.
        raw = raw.astype(float).interpolate(limit=3).dropna()

        local_year = _select_local_year(raw, offset, year)
        frame, leap = _finalise_year(local_year, utc_offset_hours=offset, year=year)
        return WeatherYear(
            data=frame,
            source=self.name,
            latitude=latitude,
            longitude=longitude,
            timezone_name=tz_name,
            utc_offset_hours=offset,
            retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            year=year,
            dataset=(
                f"ERA5 reanalysis via Open-Meteo archive "
                f"(grid {payload.get('latitude')}, {payload.get('longitude')}, "
                f"elevation {payload.get('elevation')} m)"
            ),
            leap_day_dropped=leap,
            notes=(
                "Reanalysis, not a satellite cloud retrieval. Smooths sub-grid "
                "cloud structure and tends to over-predict irradiance under "
                "broken cloud, which understates solar droughts and therefore "
                "understates the value of flexible control."
            ),
        )


HISTORICAL_PROVIDERS: dict[str, HistoricalWeatherProvider] = {
    "open_meteo_era5": OpenMeteoHistoricalProvider(),
    "nsrdb_psm4_conus": NSRDBHistoricalProvider(),
}

DEFAULT_HISTORICAL_SOURCE = "open_meteo_era5"


def _historical_cache_paths(
    source: str, latitude: float, longitude: float, year: int
) -> tuple[Path, Path]:
    stem = f"{source}_{year}_{latitude:.4f}_{longitude:.4f}"
    return CACHE_DIR / f"{stem}.parquet", CACHE_DIR / f"{stem}.json"


def get_weather_year(
    latitude: float,
    longitude: float,
    year: int,
    source: str = DEFAULT_HISTORICAL_SOURCE,
    *,
    use_cache: bool = True,
    refresh: bool = False,
) -> WeatherYear:
    """Fetch (or load from cache) one actual historical year for a site."""
    if source not in HISTORICAL_PROVIDERS:
        raise ValueError(
            f"Unknown historical weather source '{source}'. "
            f"Options: {sorted(HISTORICAL_PROVIDERS)}"
        )

    data_path, meta_path = _historical_cache_paths(source, latitude, longitude, year)
    if use_cache and not refresh and data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return WeatherYear(data=pd.read_parquet(data_path), **meta)

    record = HISTORICAL_PROVIDERS[source].fetch_year(latitude, longitude, year)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        record.data.to_parquet(data_path)
        meta_path.write_text(json.dumps(record.provenance(), indent=2) + "\n")
    return record
