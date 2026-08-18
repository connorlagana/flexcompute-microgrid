#!/usr/bin/env python
"""Experiment A across every real weather year: fixed infrastructure, six policies.

    python scripts/run_multiyear_experiment_a.py
    python scripts/run_multiyear_experiment_a.py --years 2015 2019 --scales 0.4 0.2
    python scripts/run_multiyear_experiment_a.py --cooling-fixed-fraction 0.3
    python scripts/run_multiyear_experiment_a.py --curve h100_llama3_pretrain_drawaxis_sensitivity

Infrastructure scale means: 1.0 is 100% of the reference plant's solar MW,
battery MW *and* battery MWh; 0.2 is 20% of each. The GPU fleet is unchanged at
every scale. **It is not a claim that any scale is economically preferable** —
that question belongs to Experiment B and nothing here answers it.

Compute is never reported without involuntary shortfall beside it. A fixed load
can score well on compute purely by browning out, and at some scales it does.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from flexcompute.experiments import COMPARISON_LADDER, Sizing  # noqa: E402
from flexcompute.forecast import REFERENCE_NRMSE_24H_PCT  # noqa: E402
from flexcompute.gpu import DEFAULT_CURVE_NAME, get_curve  # noqa: E402
from flexcompute.multiyear import (  # noqa: E402
    DALLAS_STUDY_YEARS,
    Job,
    YearSet,
    aggregate,
    concentration_of_advantage,
    default_workers,
    run_jobs,
    year_summary,
)
from flexcompute.scenario import LOCATIONS, Scenario  # noqa: E402
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DEFAULT_SCALES = (1.00, 0.60, 0.40, 0.25, 0.20)

#: Reported for every strategy at every scale, per the study specification.
REPORTED = (
    "compute_units",
    "voluntary_throttle_mwh",
    "involuntary_shortfall_mwh",
    "hours_throttled",
    "hours_parked",
    "soc_min_mwh",
    "solar_curtailed_pct",
    "battery_cycles_per_year",
    "battery_discharged_mwh",
)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--years", type=int, nargs="+", default=list(DALLAS_STUDY_YEARS))
    p.add_argument("--source", default="open_meteo_era5")
    p.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    p.add_argument("--strategies", nargs="+", default=list(COMPARISON_LADDER))
    p.add_argument("--cooling-fixed-fraction", type=float, default=0.0)
    p.add_argument("--aggregation", default="time_shared",
                   choices=["time_shared", "per_device"])
    p.add_argument("--curve", default=DEFAULT_CURVE_NAME)
    p.add_argument("--forecast-nrmse-pct", type=float, default=REFERENCE_NRMSE_24H_PCT)
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

    # The reference plant: the fixed-load-optimal design from the TMY baseline.
    # Held constant across every year so that "scale 0.40" means one plant.
    snapshot_scenario = Scenario(location=args.location)
    s = load_snapshot(SNAPSHOT_DIR / f"{snapshot_scenario.label()}.json")["optimized"]["sizing"]
    reference = Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"])

    print(f"location        : {args.location}")
    print(f"years           : {len(year_set.years)} x {args.source} "
          f"({min(year_set.years)}-{max(year_set.years)})")
    print(f"reference plant : {reference.solar_mw:.1f} MW-DC, "
          f"{reference.battery_mw:.1f} MW / {reference.battery_mwh:.0f} MWh")
    print(f"GPU curve       : {curve.provenance.name} [{curve.provenance.kind}]")
    for warning in curve.basis_warnings():
        print(f"   caveat       : {warning}")
    print(f"cooling fixed   : {args.cooling_fixed_fraction:.2f}    "
          f"aggregation: {args.aggregation}")
    print(f"forecast target : {args.forecast_nrmse_pct:.1f}% realised day-ahead nRMSE "
          f"(an error magnitude, not a failure rate)")
    print(f"workers         : {args.workers}")
    print()

    run_kwargs = dict(
        curve=curve,
        aggregation=args.aggregation,
        cooling_fixed_fraction=args.cooling_fixed_fraction,
        mpc_horizon_hours=args.mpc_horizon_hours,
        forecast_nrmse_pct=args.forecast_nrmse_pct,
    )

    jobs = []
    for scenario in year_set.scenarios():
        for scale in args.scales:
            sizing = reference.scaled(scale)
            for strategy in args.strategies:
                jobs.append(Job(
                    scenario=scenario, strategy=strategy,
                    solar_mw=sizing.solar_mw, battery_mw=sizing.battery_mw,
                    duration_h=sizing.duration_h,
                    run_kwargs=dict(run_kwargs), tag=(round(scale, 4),),
                ))

    started = time.time()
    total = len(jobs)
    print(f"running {total} simulations ...", flush=True)

    def progress(done: int, n: int, key) -> None:
        if done % 10 == 0 or done == n:
            elapsed = time.time() - started
            rate = done / elapsed if elapsed else 0
            eta = (n - done) / rate if rate else 0
            print(f"  {done:>4}/{n}  {elapsed/60:5.1f} min elapsed, "
                  f"{eta/60:5.1f} min remaining", flush=True)

    raw = run_jobs(jobs, workers=args.workers, progress=progress)
    print(f"done in {(time.time() - started)/60:.1f} min\n")

    # -- weather description of each year (cheap, and needed to read the table)
    weather = {}
    for scenario in year_set.scenarios():
        weather[scenario.weather_year] = year_summary(scenario.build())

    report = _assemble(raw, year_set.years, args.scales, args.strategies)
    _print_report(report, year_set.years, args.scales, args.strategies, weather)

    payload = {
        "experiment": "A_multiyear",
        "year_set": year_set.metadata(),
        "reference_sizing": reference.as_dict(),
        "scales": list(args.scales),
        "strategies": list(args.strategies),
        "settings": {
            "curve": curve.provenance.name,
            "curve_kind": curve.provenance.kind,
            "curve_basis_warnings": curve.basis_warnings(),
            "aggregation": args.aggregation,
            "cooling_fixed_fraction": args.cooling_fixed_fraction,
            "mpc_horizon_hours": args.mpc_horizon_hours,
            "forecast_target_nrmse_pct": args.forecast_nrmse_pct,
        },
        "scale_meaning": (
            "Fraction of the reference plant's solar MW, battery MW and battery "
            "MWh, with the same 10,000-GPU fleet. NOT a claim about economically "
            "preferable sizing; see Experiment B."
        ),
        "weather": weather,
        "per_year": report["per_year"],
        "aggregates": report["aggregates"],
        "concentration": report["concentration"],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / (
        f"experiment_a_multiyear_{year_set.label}"
        f"_cf{args.cooling_fixed_fraction:.2f}_{curve.provenance.name}.json"
    )
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    print(f"\nWrote {out}")
    return 0


def _assemble(raw, years, scales, strategies) -> dict:
    per_year: dict = {}
    for year in years:
        per_year[str(year)] = {}
        for scale in scales:
            key_scale = str(round(scale, 4))
            per_year[str(year)][key_scale] = {}
            baseline = raw.get((year, "fixed_load", round(scale, 4)), {})
            base_compute = baseline.get("compute_units")
            for strategy in strategies:
                metrics = raw.get((year, strategy, round(scale, 4)))
                if metrics is None:
                    continue
                row = {k: metrics.get(k) for k in REPORTED}
                row["advantage_pct_vs_fixed"] = (
                    100.0 * (metrics["compute_units"] / base_compute - 1.0)
                    if base_compute else float("nan")
                )
                for extra in ("forecast_nrmse_24h_pct_of_capacity",
                              "forecast_target_nrmse_pct", "forecast_sigma_24h",
                              "cyclic_soc_residual_mwh"):
                    if extra in metrics:
                        row[extra] = metrics[extra]
                per_year[str(year)][key_scale][strategy] = row

    aggregates: dict = {}
    concentration: dict = {}
    for scale in scales:
        key_scale = str(round(scale, 4))
        aggregates[key_scale] = {}
        concentration[key_scale] = {}
        for strategy in strategies:
            rows = [
                per_year[str(y)][key_scale][strategy]
                for y in years
                if strategy in per_year[str(y)][key_scale]
            ]
            if not rows:
                continue
            aggregates[key_scale][strategy] = {
                metric: aggregate(r[metric] for r in rows if r.get(metric) is not None)
                for metric in list(REPORTED) + ["advantage_pct_vs_fixed"]
            }
            if strategy != "fixed_load":
                by_year = {
                    y: per_year[str(y)][key_scale][strategy]["advantage_pct_vs_fixed"]
                    for y in years
                    if strategy in per_year[str(y)][key_scale]
                }
                concentration[key_scale][strategy] = concentration_of_advantage(by_year)
    return {"per_year": per_year, "aggregates": aggregates,
            "concentration": concentration}


def _print_report(report, years, scales, strategies, weather) -> None:
    print("=" * 110)
    print("EXPERIMENT A — same plant, six policies, across actual weather years")
    print("=" * 110)

    for scale in scales:
        key = str(round(scale, 4))
        print(f"\n--- infrastructure scale {scale:.2f} "
              f"(same {len(years)}-year set, same 10,000 GPUs) ---")
        print(f"{'strategy':<26}{'compute (median)':>18}{'advantage vs fixed':>30}"
              f"{'shortfall MWh (median)':>24}")
        print(f"{'':26}{'':>18}{'median':>10}{'P10':>10}{'P90':>10}{'':>24}")
        print("-" * 110)
        for strategy in strategies:
            agg = report["aggregates"].get(key, {}).get(strategy)
            if not agg:
                continue
            comp = agg["compute_units"]
            adv = agg["advantage_pct_vs_fixed"]
            short = agg["involuntary_shortfall_mwh"]
            print(f"{strategy:<26}{comp['median']:>18.1f}"
                  f"{adv['median']:>+9.2f}%{adv['p10']:>+9.2f}%{adv['p90']:>+9.2f}%"
                  f"{short['median']:>24.1f}")

        conc = report["concentration"].get(key, {})
        interpretable = {k: c for k, c in conc.items() if c["interpretable"]}
        if interpretable:
            even = next(iter(interpretable.values()))["even_split_top_3_share"]
            print(f"\n  is the advantage consistent, or carried by a few years?")
            print(f"  {'strategy':<26}{'yrs +ve':>10}{'top-1':>8}{'top-2':>8}"
                  f"{'top-3':>8}   (an even split puts {even:.0%} in top-3)")
            for strategy, c in interpretable.items():
                print(f"  {strategy:<26}"
                      f"{c['years_with_positive_advantage']:>5}/"
                      f"{c['years_evaluated']:<4}"
                      f"{c['top_1_year_share']:>8.0%}{c['top_2_year_share']:>8.0%}"
                      f"{c['top_3_year_share']:>8.0%}")
            for strategy, c in conc.items():
                if not c["interpretable"]:
                    print(f"  {strategy:<26}   no positive advantage in any year "
                          f"— concentration undefined")

    _check_ceiling(report, years, scales, strategies)

    print("\n" + "-" * 110)
    print("Infrastructure scale is an Experiment A device only. It does not say "
          "any scale is economically\npreferable — Experiment B decides sizing.")


def _check_ceiling(report, years, scales, strategies) -> None:
    """No strategy may beat perfect foresight without paying in shortfall.

    Run on every result rather than spot-checked, because it is the one
    invariant that would let a broken controller look like a discovery. A
    receding-horizon MPC *can* legitimately edge past the annual ceiling by
    ending the year with an empty battery -- it truncates at 31 December and
    never repays the energy -- and that shows up here as an exceedance with
    non-zero shortfall (ASSUMPTIONS B11). An exceedance with *zero* shortfall
    would be free energy, and is a bug.
    """
    if "perfect_foresight_annual" not in strategies:
        return
    paid, free = [], []
    for year in years:
        for scale in scales:
            rows = report["per_year"][str(year)][str(round(scale, 4))]
            ceiling = rows.get("perfect_foresight_annual")
            if ceiling is None:
                continue
            for name, row in rows.items():
                if name == "perfect_foresight_annual":
                    continue
                excess = row["compute_units"] - ceiling["compute_units"]
                if excess > 1e-6:
                    entry = (year, scale, name, excess,
                             row["involuntary_shortfall_mwh"])
                    (paid if row["involuntary_shortfall_mwh"] > 1e-6 else free).append(entry)

    print("\n" + "-" * 110)
    print("INVARIANT — nothing beats perfect foresight for free")
    print(f"  runs checked                      : {len(years) * len(scales) * len(strategies)}")
    print(f"  exceedances paid for in shortfall : {len(paid)}"
          f"   (legitimate; see ASSUMPTIONS B11)")
    print(f"  exceedances with ZERO shortfall    : {len(free)}")
    if free:
        print("  *** FREE ENERGY — this is a bug, not a result ***")
        for year, scale, name, excess, _ in free[:10]:
            print(f"      {year} scale {scale:.2f} {name}: +{excess:.4f} compute-unit-hours")
    elif paid:
        worst = max(paid, key=lambda e: e[3])
        names = sorted({e[2] for e in paid})
        print(f"  largest                            : {worst[2]} in {worst[0]} at scale "
              f"{worst[1]:.2f}, +{worst[3]:.2f} cu-h for {worst[4]:.1f} MWh unserved")
        print(f"  strategies involved                : {', '.join(names)}")


if __name__ == "__main__":
    raise SystemExit(main())
