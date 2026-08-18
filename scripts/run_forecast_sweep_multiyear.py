#!/usr/bin/env python
"""What a wrong forecast costs, across every real weather year.

    python scripts/run_forecast_sweep_multiyear.py
    python scripts/run_forecast_sweep_multiyear.py --scale 0.25 --nrmse 5 10 15 20

The sweep is indexed by **realised day-ahead nRMSE**, not by the error model's
internal sigma. That matters across years: the same sigma lands at a different
realised error in a cloudy year than in a sunny one, so a sigma-indexed sweep
would compare different error levels while labelling them the same. Each year is
calibrated independently to hit the requested realised error, and the achieved
error is recorded alongside every result.

nRMSE here is an error *magnitude* normalised by plant capacity. It is not a
fraction of hours forecast wrongly, and must never be described that way.

Two reference lines bound the achievable range: ``fixed_load`` (no control) and
``perfect_foresight_annual`` (the ceiling). Without them a two-percent spread
looks either trivial or enormous depending on the axis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.experiments import Sizing            # noqa: E402
from flexcompute.forecast import NRMSE_SWEEP_PCT      # noqa: E402
from flexcompute.gpu import DEFAULT_CURVE_NAME, get_curve  # noqa: E402
from flexcompute.multiyear import (                   # noqa: E402
    DALLAS_STUDY_YEARS,
    Job,
    YearSet,
    aggregate,
    default_workers,
    run_jobs,
)
from flexcompute.scenario import LOCATIONS, Scenario  # noqa: E402
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--years", type=int, nargs="+", default=list(DALLAS_STUDY_YEARS))
    p.add_argument("--source", default="open_meteo_era5")
    p.add_argument("--scale", type=float, default=0.25,
                   help="infrastructure scale; pick one where stored energy binds")
    p.add_argument("--nrmse", type=float, nargs="+", default=list(NRMSE_SWEEP_PCT))
    p.add_argument("--cooling-fixed-fraction", type=float, default=0.0)
    p.add_argument("--curve", default=DEFAULT_CURVE_NAME)
    p.add_argument("--mpc-horizon-hours", type=int, default=48)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--out", default=None)
    args = p.parse_args()

    logging.disable(logging.INFO)
    curve = get_curve(args.curve)
    year_set = YearSet(
        years=tuple(sorted(args.years)), source=args.source, location=args.location,
        label=f"{args.location}_{min(args.years)}_{max(args.years)}_{args.source}",
    )

    tmy_scenario = Scenario(location=args.location)
    s = load_snapshot(SNAPSHOT_DIR / f"{tmy_scenario.label()}.json")["optimized"]["sizing"]
    sizing = Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"]).scaled(args.scale)

    print(f"scale {args.scale:.2f}  ->  {sizing.solar_mw:.1f} MW-DC, "
          f"{sizing.battery_mw:.1f} MW / {sizing.battery_mwh:.0f} MWh")
    print(f"years  : {len(year_set.years)} x {args.source}")
    print(f"curve  : {curve.provenance.name} [{curve.provenance.kind}]")
    print(f"sweep  : {args.nrmse} % realised day-ahead nRMSE "
          f"(error magnitude, not a failure rate)\n")

    common = dict(
        curve=curve,
        cooling_fixed_fraction=args.cooling_fixed_fraction,
        mpc_horizon_hours=args.mpc_horizon_hours,
    )

    jobs = []
    for scenario in year_set.scenarios():
        for reference in ("fixed_load", "perfect_foresight_annual"):
            jobs.append(Job(scenario=scenario, strategy=reference,
                            solar_mw=sizing.solar_mw, battery_mw=sizing.battery_mw,
                            duration_h=sizing.duration_h,
                            run_kwargs=dict(common), tag=("reference",)))
        for target in args.nrmse:
            jobs.append(Job(scenario=scenario, strategy="forecast_mpc",
                            solar_mw=sizing.solar_mw, battery_mw=sizing.battery_mw,
                            duration_h=sizing.duration_h,
                            run_kwargs=dict(common, forecast_nrmse_pct=target),
                            tag=(round(float(target), 3),)))

    started = time.time()
    print(f"running {len(jobs)} simulations ...", flush=True)

    def progress(done: int, n: int, key) -> None:
        if done % 10 == 0 or done == n:
            elapsed = time.time() - started
            eta = (n - done) * elapsed / max(done, 1)
            print(f"  {done:>4}/{n}  {elapsed/60:5.1f} min elapsed, "
                  f"{eta/60:5.1f} min remaining", flush=True)

    raw = run_jobs(jobs, workers=args.workers, progress=progress)
    print(f"done in {(time.time() - started)/60:.1f} min\n")

    per_year: dict = {}
    for year in year_set.years:
        entry = {
            "fixed_load": raw[(year, "fixed_load", "reference")]["compute_units"],
            "perfect_foresight_annual":
                raw[(year, "perfect_foresight_annual", "reference")]["compute_units"],
            "forecast_mpc": {},
        }
        for target in args.nrmse:
            m = raw[(year, "forecast_mpc", round(float(target), 3))]
            entry["forecast_mpc"][str(target)] = {
                "compute_units": m["compute_units"],
                "involuntary_shortfall_mwh": m["involuntary_shortfall_mwh"],
                "voluntary_throttle_mwh": m["voluntary_throttle_mwh"],
                "realised_nrmse_24h_pct": m.get("forecast_nrmse_24h_pct_of_capacity"),
                "target_nrmse_pct": m.get("forecast_target_nrmse_pct"),
                "sigma_24h": m.get("forecast_sigma_24h"),
            }
        per_year[str(year)] = entry

    rows = []
    for target in args.nrmse:
        key = str(target)
        computes = [per_year[str(y)]["forecast_mpc"][key]["compute_units"]
                    for y in year_set.years]
        realised = [per_year[str(y)]["forecast_mpc"][key]["realised_nrmse_24h_pct"]
                    for y in year_set.years]
        shortfall = [per_year[str(y)]["forecast_mpc"][key]["involuntary_shortfall_mwh"]
                     for y in year_set.years]
        rows.append({
            "target_nrmse_pct": float(target),
            "realised_nrmse": aggregate(realised),
            "compute": aggregate(computes),
            "involuntary_shortfall_mwh": aggregate(shortfall),
        })

    references = {
        "fixed_load": aggregate(per_year[str(y)]["fixed_load"] for y in year_set.years),
        "perfect_foresight_annual": aggregate(
            per_year[str(y)]["perfect_foresight_annual"] for y in year_set.years),
    }

    _report(rows, references, args.scale, len(year_set.years))

    payload = {
        "experiment": "forecast_error_sweep_multiyear",
        "year_set": year_set.metadata(),
        "sizing": sizing.as_dict(),
        "infrastructure_scale": args.scale,
        "settings": {
            "curve": curve.provenance.name,
            "curve_kind": curve.provenance.kind,
            "curve_basis_warnings": curve.basis_warnings(),
            "cooling_fixed_fraction": args.cooling_fixed_fraction,
            "mpc_horizon_hours": args.mpc_horizon_hours,
        },
        "nrmse_is": (
            "Realised day-ahead nRMSE, an error magnitude normalised by plant "
            "capacity, scored over daylight hours. NOT a fraction of hours "
            "forecast wrongly."
        ),
        "caveat": (
            "The forecast error model is synthetic and not a validated "
            "forecasting system (ASSUMPTIONS B13). Read the shape of the "
            "response, not the value at one point."
        ),
        "references": references,
        "rows": rows,
        "per_year": per_year,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / (
        f"forecast_sweep_multiyear_{year_set.label}_scale{args.scale:.2f}.json")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    print(f"\nWrote {out}")
    return 0


def _report(rows, references, scale, n_years) -> None:
    fixed = references["fixed_load"]["median"]
    ceiling = references["perfect_foresight_annual"]["median"]
    window = ceiling - fixed

    print("=" * 96)
    print(f"FORECAST ERROR SWEEP — infrastructure scale {scale:.2f}, "
          f"median of {n_years} weather years")
    print("=" * 96)
    print(f"{'target nRMSE':>13}{'realised':>11}{'compute':>12}{'vs fixed':>11}"
          f"{'advantage kept':>17}{'shortfall MWh':>16}")
    print("-" * 96)
    print(f"{'perfect (ceiling)':>13}{'—':>11}{ceiling:>12.1f}"
          f"{100*(ceiling/fixed-1):>+10.2f}%{'100%':>17}{0.0:>16.1f}")
    for row in rows:
        compute = row["compute"]["median"]
        kept = 100.0 * (compute - fixed) / window if window else float("nan")
        print(f"{row['target_nrmse_pct']:>12.0f}%"
              f"{row['realised_nrmse']['median']:>10.2f}%"
              f"{compute:>12.1f}{100*(compute/fixed-1):>+10.2f}%"
              f"{kept:>16.1f}%"
              f"{row['involuntary_shortfall_mwh']['median']:>16.1f}")
    print(f"{'no control':>13}{'—':>11}{fixed:>12.1f}{0.0:>+10.2f}%{'0%':>17}"
          f"{'(see JSON)':>16}")
    print("\n  'advantage kept' is the share of the perfect-foresight prize that")
    print("  survives not knowing the weather. The prize itself is only "
          f"{window:,.0f} compute-unit-hours,\n  so read the absolute column too.")


if __name__ == "__main__":
    raise SystemExit(main())
