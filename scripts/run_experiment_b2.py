#!/usr/bin/env python
"""Experiment B: minimum year-0 CAPEX at equal compute, and at equal reliability.

    python scripts/run_experiment_b2.py                      # B1 + B2, free duration
    python scripts/run_experiment_b2.py --durations 4 8 12 free
    python scripts/run_experiment_b2.py --cooling-fixed-fraction 0.3
    python scripts/run_experiment_b2.py --variants B2 --strategies fixed_load casey_governor

Two variants, reported separately, because they answer different questions.

**B1 — equal compute only.** Preserved for continuity with the earlier result.
Its weakness is that a design may hit the compute target by browning out: the
re-sized fixed-load plant books thousands of MWh of involuntary shortfall a year
while the flexible designs book none, so the comparison is not like-for-like.

**B2 — equal compute *and* equal reliability.** Every strategy must also keep
annual involuntary shortfall at or below a common absolute cap. The cap is
derived from the reference plant's own behaviour rather than chosen to suit any
strategy: it is what the 99%-uptime-designed reference plant already delivers
under a fixed load. A flexible controller must buy its reliability out of the
same capital budget as everyone else, and a design that was already reliable
pays nothing for the constraint — so the criterion does not favour flexibility
by construction.

The headline infrastructure-saving number should come from B2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.costs import from_upstream_config, upstream_equivalent_capex_usd  # noqa: E402
from flexcompute.experiments import (          # noqa: E402
    COST_SOURCE_DURATION_RANGE_H,
    Sizing,
    SizingSearchSpec,
    run_sizing_searches,
    run_strategy,
)
from flexcompute.forecast import REFERENCE_NRMSE_24H_PCT  # noqa: E402
from flexcompute.gpu import DEFAULT_CURVE_NAME, get_curve  # noqa: E402
from flexcompute.multiyear import default_workers  # noqa: E402
from flexcompute.scenario import LOCATIONS, Scenario  # noqa: E402
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DEFAULT_STRATEGIES = ("fixed_load", "casey_governor", "forecast_mpc",
                      "perfect_foresight_annual")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--weather-year", type=int, default=None,
                   help="actual calendar year; omit for the TMY")
    p.add_argument("--source", default="open_meteo_era5")
    p.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    p.add_argument("--variants", nargs="+", default=["B1", "B2"], choices=["B1", "B2"])
    p.add_argument("--durations", nargs="+", default=["free"],
                   help="'free' or fixed hours, e.g. --durations 4 8 12 free")
    p.add_argument("--cooling-fixed-fraction", type=float, default=0.0)
    p.add_argument("--aggregation", default="time_shared",
                   choices=["time_shared", "per_device"])
    p.add_argument("--curve", default=DEFAULT_CURVE_NAME)
    p.add_argument("--forecast-nrmse-pct", type=float, default=REFERENCE_NRMSE_24H_PCT)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--popsize", type=int, default=10)
    # A receding-horizon MPC costs ~45 s per simulated year against ~1 s for
    # every other strategy, so a full-budget search for it would take days. It
    # gets a smaller budget and only the free-duration case. A less thorough
    # search can only fail to find a cheaper design, never invent one, so this
    # biases the MPC's capex *upward* -- against the hypothesis -- and the
    # budget is recorded with every result.
    p.add_argument("--mpc-maxiter", type=int, default=10)
    p.add_argument("--mpc-popsize", type=int, default=6)
    p.add_argument("--mpc-durations", nargs="+", default=["free"],
                   help="duration cases the expensive MPC strategies are run for")
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    logging.disable(logging.INFO)
    curve = get_curve(args.curve)
    scenario = Scenario(
        location=args.location,
        weather_year=args.weather_year,
        historical_weather_source=args.source,
    )
    site = scenario.build()

    # Reference plant from the TMY baseline snapshot: the 99%-uptime design.
    tmy_scenario = Scenario(location=args.location)
    s = load_snapshot(SNAPSHOT_DIR / f"{tmy_scenario.label()}.json")["optimized"]["sizing"]
    reference = Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"])

    cost_model = from_upstream_config(site.config, scenario.architecture)
    run_kwargs = dict(
        curve=curve,
        aggregation=args.aggregation,
        cooling_fixed_fraction=args.cooling_fixed_fraction,
        forecast_nrmse_pct=args.forecast_nrmse_pct,
    )

    ours = cost_model.total_capex_usd(
        reference.solar_mw, reference.battery_mw, reference.battery_mwh) / 1e6
    theirs = upstream_equivalent_capex_usd(
        site.config, reference.solar_mw, reference.battery_mw, scenario.architecture) / 1e6

    # The target and the reliability cap both come from the *same* run: what the
    # reference plant actually delivers under a fixed load in this weather year.
    baseline = run_strategy(site, "fixed_load", reference, **run_kwargs)
    target = baseline.metrics["compute_units"]
    reliability_cap = baseline.metrics["involuntary_shortfall_mwh"]

    print(f"scenario        : {scenario.label()}")
    print(f"reference plant : {reference.solar_mw:.1f} MW-DC, "
          f"{reference.battery_mw:.1f} MW / {reference.battery_mwh:.0f} MWh "
          f"({reference.duration_h:.0f} h)   capex {ours:,.1f} M$")
    print(f"  cost reconcil.: ours {ours:.4f} vs upstream {theirs:.4f} M$ "
          f"(diff {ours - theirs:+.2e})")
    print(f"GPU curve       : {curve.provenance.name} [{curve.provenance.kind}]")
    for warning in curve.basis_warnings():
        print(f"   caveat       : {warning}")
    print(f"cooling fixed   : {args.cooling_fixed_fraction:.2f}")
    print()
    print(f"compute target  : {target:,.2f} compute-unit-hours "
          f"(the reference plant's own fixed-load output)")
    print(f"reliability cap : {reliability_cap:,.1f} MWh/yr involuntary shortfall "
          f"(what that same plant already books)")
    print()

    #: Strategies whose every evaluation simulates a receding-horizon year.
    EXPENSIVE = {"forecast_mpc", "perfect_foresight_mpc"}

    durations = [None if d == "free" else float(d) for d in args.durations]
    mpc_durations = [None if d == "free" else float(d) for d in args.mpc_durations]
    specs = []
    for variant in args.variants:
        cap = None if variant == "B1" else reliability_cap
        for strategy in args.strategies:
            expensive = strategy in EXPENSIVE
            for duration in (mpc_durations if expensive else durations):
                mode = "free" if duration is None else f"{duration:g}h"
                specs.append(SizingSearchSpec(
                    scenario=scenario,
                    strategy=strategy,
                    target_compute=target,
                    reference=reference,
                    label=f"{variant}|{mode}|{strategy}",
                    fixed_duration_h=duration,
                    max_shortfall_mwh=cap,
                    seed=args.seed,
                    maxiter=args.mpc_maxiter if expensive else args.maxiter,
                    popsize=args.mpc_popsize if expensive else args.popsize,
                    curve_name=curve.provenance.name,
                    aggregation=args.aggregation,
                    cooling_fixed_fraction=args.cooling_fixed_fraction,
                    forecast_nrmse_pct=args.forecast_nrmse_pct,
                ))
    if any(s.strategy in EXPENSIVE for s in specs):
        print(f"NOTE: {sorted(EXPENSIVE & set(args.strategies))} search on a reduced "
              f"budget (maxiter {args.mpc_maxiter}, popsize {args.mpc_popsize}, "
              f"durations {args.mpc_durations}).\n      A smaller search can only "
              f"miss a cheaper design, never invent one, so their CAPEX is an "
              f"upper bound.\n")

    started = time.time()
    print(f"running {len(specs)} sizing searches on {args.workers} workers ...\n",
          flush=True)

    def progress(done: int, total: int, label: str, r) -> None:
        elapsed = (time.time() - started) / 60
        flag = " EXTRAPOLATED-DURATION" if r.duration_extrapolated else ""
        ok = "  *** MISSED CONSTRAINT ***" if not r.success else ""
        detail = (f"{r.capex_musd:>8.1f} M$  "
                  f"{r.sizing.solar_mw:>6.1f} MW / {r.sizing.battery_mw:>5.1f} MW / "
                  f"{r.sizing.battery_mwh:>6.0f} MWh ({r.sizing.duration_h:>4.1f} h)  "
                  f"shortfall {r.involuntary_shortfall_mwh:>8.1f}  "
                  f"{r.evaluations} evals{flag}{ok}")
        print(f"  [{done:>2}/{total}] {elapsed:5.1f} min  {label:<38}{detail}", flush=True)

    results = run_sizing_searches(specs, workers=args.workers, progress=progress)
    print(f"\ntotal {(time.time() - started)/60:.1f} min")
    _report(results, args, reference_capex=ours, reliability_cap=reliability_cap)

    payload = {
        "experiment": "B_equal_compute_and_reliability",
        "scenario": scenario.label(),
        "weather_year": scenario.weather_year,
        "weather_source": scenario.effective_weather_source,
        "reference_sizing": reference.as_dict(),
        "reference_capex_musd": ours,
        "target_compute": target,
        "reliability_cap_mwh": reliability_cap,
        "reliability_cap_basis": (
            "Involuntary shortfall booked by the reference 99%-uptime plant under "
            "a fixed load in this weather year. Applied identically to every "
            "strategy; not chosen to favour flexible operation."
        ),
        "cost_model": cost_model.metadata(),
        "cost_source_duration_range_h": list(COST_SOURCE_DURATION_RANGE_H),
        "settings": {
            "curve": curve.provenance.name,
            "curve_kind": curve.provenance.kind,
            "curve_basis_warnings": curve.basis_warnings(),
            "aggregation": args.aggregation,
            "cooling_fixed_fraction": args.cooling_fixed_fraction,
            "forecast_target_nrmse_pct": args.forecast_nrmse_pct,
            "maxiter": args.maxiter, "popsize": args.popsize, "seed": args.seed,
            "mpc_maxiter": args.mpc_maxiter, "mpc_popsize": args.mpc_popsize,
            "mpc_durations": list(args.mpc_durations),
            "mpc_budget_note": (
                "Receding-horizon strategies search on a reduced budget because "
                "each evaluation simulates a full MPC year. A smaller search can "
                "only miss a cheaper design, so their CAPEX is an upper bound and "
                "their advantage over fixed load is understated."
            ),
        },
        "results": {k: v.as_dict() for k, v in results.items()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / (
        f"experiment_b2_{scenario.label()}_cf{args.cooling_fixed_fraction:.2f}"
        f"_{curve.provenance.name}.json"
    )
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    print(f"\nWrote {out}")
    return 0


def _report(results, args, *, reference_capex: float, reliability_cap: float) -> None:
    print("\n" + "=" * 112)
    print("EXPERIMENT B — minimum year-0 solar+BESS CAPEX")
    print("=" * 112)

    for variant in args.variants:
        title = ("B1 — equal compute only"
                 if variant == "B1" else
                 f"B2 — equal compute AND shortfall <= {reliability_cap:,.0f} MWh/yr")
        print(f"\n### {title}")
        for duration in args.durations:
            mode = "free" if duration == "free" else f"{float(duration):g}h"
            rows = {k.split("|")[2]: v for k, v in results.items()
                    if k.startswith(f"{variant}|{mode}|")}
            if not rows:
                continue
            print(f"\n  battery duration: {mode}")
            print(f"  {'design':<26}{'CAPEX M$':>10}{'vs ref':>9}{'vs fixed':>10}"
                  f"{'solar MW':>10}{'batt MW':>9}{'batt MWh':>10}{'dur h':>7}"
                  f"{'shortfall':>11}")
            print("  " + "-" * 108)
            fixed = rows.get("fixed_load")
            for name, r in rows.items():
                vs_fixed = (100 * (r.capex_musd / fixed.capex_musd - 1)
                            if fixed else float("nan"))
                mark = " !" if r.duration_extrapolated else "  "
                print(f"  {name:<26}{r.capex_musd:>10.1f}"
                      f"{100 * (r.capex_musd / reference_capex - 1):>+8.1f}%"
                      f"{vs_fixed:>+9.1f}%"
                      f"{r.sizing.solar_mw:>10.1f}{r.sizing.battery_mw:>9.1f}"
                      f"{r.sizing.battery_mwh:>10.0f}{r.sizing.duration_h:>5.1f}{mark}"
                      f"{r.involuntary_shortfall_mwh:>11.1f}")
                if not r.success:
                    print(f"  {'':26}  WARNING: constraint not met "
                          f"(compute {r.compute_units:,.1f} / target {r.target_compute:,.1f}, "
                          f"shortfall {r.involuntary_shortfall_mwh:,.1f})")

    extrapolated = [k for k, r in results.items() if r.duration_extrapolated]
    if extrapolated:
        low, high = COST_SOURCE_DURATION_RANGE_H
        print(f"\n  ! marks a battery duration outside {low:g}-{high:g} h, the range the "
              f"cost decomposition's\n    source data actually spans. Those CAPEX figures "
              f"are an EXTRAPOLATION of the\n    $/kW + $/kWh split, not a priced design. "
              f"{len(extrapolated)} of {len(results)} results are affected.")

    print("\n  Reported separately on purpose: B1 lets a design reach the compute")
    print("  target by browning out, so its 'saving' partly reflects reliability")
    print("  the flexible designs did not spend. B2 is the defensible comparison.")


if __name__ == "__main__":
    raise SystemExit(main())
