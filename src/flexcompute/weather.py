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
class TMY:
    """A typical-meteorological-year record plus its provenance."""

    data: pd.DataFrame
    source: str
    latitude: float
    longitude: float
    timezone_name: str
    utc_offset_hours: float
    retrieved_utc: str
    notes: str = ""

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


class WeatherProvider(Protocol):
    """Anything that can supply a TMY for a site."""

    name: str

    def fetch(self, latitude: float, longitude: float) -> TMY: ...


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

    def fetch(self, latitude: float, longitude: float) -> TMY:
        import pvlib

        tz_name, offset = _standard_utc_offset_hours(latitude, longitude)
        raw = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)[0]
        frame = to_local_standard_hour_index(_validate(raw), offset)
        return TMY(
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

    def fetch(self, latitude: float, longitude: float) -> TMY:
        from .upstream_bridge import ensure_upstream_importable

        ensure_upstream_importable()
        from nsrdb_loader import get_nsrdb_tmy  # type: ignore[import-not-found]

        tz_name, offset = _standard_utc_offset_hours(latitude, longitude)
        raw = get_nsrdb_tmy(latitude, longitude)
        frame = to_local_standard_hour_index(_validate(raw), offset)
        return TMY(
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
