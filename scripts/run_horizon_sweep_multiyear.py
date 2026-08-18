#!/usr/bin/env python
"""How far ahead does a controller need to see?

    python scripts/run_horizon_sweep_multiyear.py
    python scripts/run_horizon_sweep_multiyear.py --horizons 12 24 48 96 --scale 0.25

Run at a scarce sizing, where lookahead can actually matter, under **both**
perfect foresight and a realistic forecast. The two answer different questions:

* perfect foresight isolates the cost of a *finite horizon* alone;
* the forecast-aware run adds the fact that a longer horizon also means looking
  further into a belief that gets worse with lead time.

Reported as the share of the perfect-foresight *advantage* retained, not as a
percentage of the ceiling. Percent-of-ceiling divides by a number only a few
percent above doing nothing, which pins every entry near 100% and makes short
horizons look far better than they are.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.experiments import Sizing        # noqa: E402
from flexcompute.forecast import REFERENCE_NRMSE_24H_PCT  # noqa: E402
from flexcompute.gpu import DEFAULT_CURVE_NAME, get_curve  # noqa: E402
from flexcompute.multiyear import (               # noqa: E402
    Job, YearSet, aggregate, default_workers, run_jobs,
)
from flexcompute.scenario import LOCATIONS, Scenario  # noqa: E402
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

#: Three years spanning the record: the drought year, a mid-pack year, and the
#: sunniest. A full 15-year sweep would be ~4x the MPC cost for a result whose
#: spread Experiment A already shows to be small.
DEFAULT_YEARS = (2015, 2019, 2022)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    p.add_argument("--source", default="open_meteo_era5")
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 48, 96])
    p.add_argument("--curve", default=DEFAULT_CURVE_NAME)
    p.add_argument("--forecast-nrmse-pct", type=float, default=REFERENCE_NRMSE_24H_PCT)
    p.add_argument("--cooling-fixed-fraction", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--out", default=None)
    args = p.parse_args()

    logging.disable(logging.INFO)
    curve = get_curve(args.curve)
    year_set = YearSet(years=tuple(sorted(args.years)), source=args.source,
                       location=args.location,
                       label=f"{args.location}_{'_'.join(map(str, sorted(args.years)))}")

    s = load_snapshot(
        SNAPSHOT_DIR / f"{Scenario(location=args.location).label()}.json"
    )["optimized"]["sizing"]
    sizing = Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"]).scaled(args.scale)

    print(f"scale {args.scale:.2f} -> {sizing.solar_mw:.1f} MW-DC, "
          f"{sizing.battery_mw:.1f} MW / {sizing.battery_mwh:.0f} MWh")
    print(f"years   : {list(year_set.years)}")
    print(f"curve   : {curve.provenance.name} [{curve.provenance.kind}]")
    print(f"forecast: {args.forecast_nrmse_pct:.0f}% realised day-ahead nRMSE\n")

    common = dict(curve=curve, cooling_fixed_fraction=args.cooling_fixed_fraction)
    jobs = []
    for scenario in year_set.scenarios():
        for reference in ("fixed_load", "perfect_foresight_annual"):
            jobs.append(Job(scenario=scenario, strategy=reference,
                            solar_mw=sizing.solar_mw, battery_mw=sizing.battery_mw,
                            duration_h=sizing.duration_h,
                            run_kwargs=dict(common), tag=("reference",)))
        for horizon in args.horizons:
            for strategy in ("perfect_foresight_mpc", "forecast_mpc"):
                kwargs = dict(common, mpc_horizon_hours=horizon)
                if strategy == "forecast_mpc":
                    kwargs["forecast_nrmse_pct"] = args.forecast_nrmse_pct
                jobs.append(Job(scenario=scenario, strategy=strategy,
                                solar_mw=sizing.solar_mw, battery_mw=sizing.battery_mw,
                                duration_h=sizing.duration_h,
                                run_kwargs=kwargs, tag=(horizon,)))

    started = time.time()
    print(f"running {len(jobs)} simulations ...", flush=True)

    def progress(done, n, key):
        if done % 5 == 0 or done == n:
            e = time.time() - started
            print(f"  {done:>3}/{n}  {e/60:5.1f} min elapsed, "
                  f"{(n-done)*e/max(done,1)/60:5.1f} min remaining", flush=True)

    raw = run_jobs(jobs, workers=args.workers, progress=progress)
    print(f"done in {(time.time() - started)/60:.1f} min\n")

    fixed = aggregate(raw[(y, "fixed_load", "reference")]["compute_units"]
                      for y in year_set.years)["median"]
    ceiling = aggregate(raw[(y, "perfect_foresight_annual", "reference")]["compute_units"]
                        for y in year_set.years)["median"]
    window = ceiling - fixed

    rows = []
    for horizon in args.horizons:
        row = {"horizon_hours": horizon}
        for strategy in ("perfect_foresight_mpc", "forecast_mpc"):
            compute = aggregate(raw[(y, strategy, horizon)]["compute_units"]
                                for y in year_set.years)
            short = aggregate(raw[(y, strategy, horizon)]["involuntary_shortfall_mwh"]
                              for y in year_set.years)
            row[strategy] = {
                "compute": compute,
                "involuntary_shortfall_mwh": short,
                "advantage_retained_pct":
                    100.0 * (compute["median"] - fixed) / window if window else float("nan"),
            }
        rows.append(row)

    print("=" * 100)
    print(f"HORIZON SWEEP — scale {args.scale:.2f}, median of {len(year_set.years)} years")
    print(f"perfect foresight is worth {window:,.0f} compute-unit-hours over no control")
    print("=" * 100)
    print(f"{'lookahead':>10}{'perfect foresight':>28}{'forecast-aware':>28}{'unserved':>12}")
    print(f"{'':>10}{'compute':>14}{'retained':>14}{'compute':>14}{'retained':>14}{'MWh':>12}")
    print("-" * 100)
    for row in rows:
        pf, fc = row["perfect_foresight_mpc"], row["forecast_mpc"]
        print(f"{row['horizon_hours']:>8} h"
              f"{pf['compute']['median']:>14.1f}{pf['advantage_retained_pct']:>13.1f}%"
              f"{fc['compute']['median']:>14.1f}{fc['advantage_retained_pct']:>13.1f}%"
              f"{fc['involuntary_shortfall_mwh']['median']:>12.1f}")
    print(f"{'annual LP':>10}{ceiling:>14.1f}{100.0:>13.1f}%{'—':>14}{'—':>14}{0.0:>12.1f}")
    print(f"{'no control':>10}{fixed:>14.1f}{0.0:>13.1f}%{'—':>14}{'—':>14}")

    payload = {
        "experiment": "horizon_sweep_multiyear",
        "year_set": year_set.metadata(),
        "sizing": sizing.as_dict(),
        "infrastructure_scale": args.scale,
        "settings": {"curve": curve.provenance.name,
                     "curve_kind": curve.provenance.kind,
                     "forecast_target_nrmse_pct": args.forecast_nrmse_pct,
                     "cooling_fixed_fraction": args.cooling_fixed_fraction},
        "references": {"fixed_load_median": fixed,
                       "perfect_foresight_annual_median": ceiling,
                       "advantage_window": window},
        "metric_note": (
            "'retained' is the share of the perfect-foresight advantage, not a "
            "percentage of the ceiling. Percent-of-ceiling flatters short "
            "horizons because the ceiling sits only a few percent above doing "
            "nothing."
        ),
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / (
        f"horizon_sweep_multiyear_{year_set.label}_scale{args.scale:.2f}.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
