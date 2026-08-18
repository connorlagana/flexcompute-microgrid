#!/usr/bin/env python
"""Download and cache actual historical weather years for a site.

    python scripts/fetch_weather_years.py                       # Dallas 2010-2024, ERA5
    python scripts/fetch_weather_years.py --source nsrdb_psm4_conus --years 2018 2024
    python scripts/fetch_weather_years.py --check               # report what is cached

Two sources, and they are not interchangeable:

``open_meteo_era5``
    ERA5 reanalysis. Covers 1940-present, no API key. The only source able to
    supply a full 15-year window here. Smooths sub-grid cloud, so it slightly
    understates solar droughts — a bias that runs *against* the value of
    flexible control, which is why it is acceptable as the primary source.

``nsrdb_psm4_conus``
    NSRDB PSM v4, GOES satellite retrieval. The reference paper's source and
    the better irradiance record, but the v4 CONUS product starts in 2018.
    Needs ``NLR_API_KEY`` and ``NLR_EMAIL``; ``DEMO_KEY`` is rate limited to a
    few requests an hour and will fail partway through a bulk fetch.

Do not mix sources across years within one study. A change of source between
2017 and 2018 would appear in the results as weather variation when it is
really an instrumentation change. Run each source over its own complete window
and compare the windows, which is what ``--check`` is for.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.scenario import LOCATIONS                      # noqa: E402
from flexcompute.weather import (                               # noqa: E402
    HISTORICAL_PROVIDERS,
    NSRDBHistoricalProvider,
    _historical_cache_paths,
    get_weather_year,
)

DEFAULT_YEARS = tuple(range(2010, 2025))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--source", default="open_meteo_era5",
                   choices=sorted(HISTORICAL_PROVIDERS))
    p.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"),
                   default=(DEFAULT_YEARS[0], DEFAULT_YEARS[-1]))
    p.add_argument("--sleep", type=float, default=1.0,
                   help="seconds between requests, to stay inside rate limits")
    p.add_argument("--check", action="store_true",
                   help="report cache status and exit without fetching")
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    latitude, longitude = LOCATIONS[args.location]
    years = list(range(args.years[0], args.years[1] + 1))
    provider = HISTORICAL_PROVIDERS[args.source]
    first, last = provider.available_years()

    print(f"site   : {args.location} ({latitude:.2f}, {longitude:.2f})")
    print(f"source : {args.source}  (covers {first}-{last})")
    if args.source == "nsrdb_psm4_conus" and not NSRDBHistoricalProvider.has_credentials():
        print("  NOTE: NLR_API_KEY / NLR_EMAIL not set — falling back to DEMO_KEY,")
        print("        which is rate limited to a few requests per hour.")
    print()

    out_of_range = [y for y in years if not first <= y <= last]
    if out_of_range:
        print(f"  {args.source} cannot supply {out_of_range}.")
        if not args.check:
            print("  Refusing to silently substitute another source for those years:")
            print("  a source change inside one study reads as weather variation.")
            return 1

    failures = []
    for year in years:
        data_path, _ = _historical_cache_paths(args.source, latitude, longitude, year)
        cached = data_path.exists()
        if args.check:
            print(f"  {year}  {'cached' if cached else 'MISSING'}")
            continue
        if cached and not args.refresh:
            print(f"  {year}  cached")
            continue
        if year not in range(first, last + 1):
            continue
        try:
            record = get_weather_year(
                latitude, longitude, year, args.source, refresh=args.refresh
            )
            s = record.summary()
            print(f"  {year}  GHI {s['ghi_annual_kwh_per_m2']:7.1f} kWh/m2   "
                  f"DNI {s['dni_annual_kwh_per_m2']:7.1f}   "
                  f"mean T {s['temp_air_mean_c']:5.1f} C   "
                  f"{'(leap day dropped)' if record.leap_day_dropped else ''}")
        except Exception as exc:                      # noqa: BLE001
            print(f"  {year}  FAILED: {type(exc).__name__}: {exc}")
            failures.append(year)
        time.sleep(args.sleep)

    if failures:
        print(f"\n{len(failures)} year(s) failed: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
