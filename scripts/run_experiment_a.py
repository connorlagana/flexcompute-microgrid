#!/usr/bin/env python
"""Experiment A: fixed infrastructure, varying control policy.

Every strategy gets identical weather, hardware, demand and starting
conditions. The only difference is who decides GPU power.

    python scripts/run_experiment_a.py                 # baseline sizing + figure
    python scripts/run_experiment_a.py --derate-sweep  # + shrink the plant
    python scripts/run_experiment_a.py --fast          # skip the MPC (seconds)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.experiments import (            # noqa: E402
    STRATEGY_ORDER,
    Sizing,
    derate_sweep,
    infrastructure_for_target_compute,
    run_experiment_a,
)
from flexcompute.gpu import get_curve            # noqa: E402
from flexcompute.plotting import plot_derate_sweep, plot_difficult_window  # noqa: E402
from flexcompute.scenario import LOCATIONS, Scenario    # noqa: E402
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_SCALES = (1.00, 0.80, 0.60, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20)


def baseline_sizing(scenario: Scenario) -> Sizing:
    path = SNAPSHOT_DIR / f"{scenario.label()}.json"
    if not path.exists():
        raise SystemExit(f"No baseline snapshot at {path}. Run scripts/run_baseline.py first.")
    s = load_snapshot(path)["optimized"]["sizing"]
    return Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--gpus", type=int, default=10_000)
    p.add_argument("--curve", default=None)
    p.add_argument("--aggregation", default="time_shared", choices=["time_shared", "per_device"])
    p.add_argument("--mpc-horizon", type=int, default=48)
    p.add_argument("--terminal-value-scale", type=float, default=0.95)
    p.add_argument("--window-hours", type=int, default=72)
    p.add_argument("--derate-sweep", action="store_true")
    p.add_argument("--fast", action="store_true", help="Skip the receding-horizon MPC.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if not args.verbose:
        logging.disable(logging.INFO)

    scenario = Scenario(location=args.location, total_gpus=args.gpus)
    site = scenario.build()
    sizing = baseline_sizing(scenario)
    curve = get_curve(args.curve) if args.curve else get_curve()

    strategies = tuple(s for s in STRATEGY_ORDER
                       if not (args.fast and s == "perfect_foresight_mpc"))
    kwargs = dict(
        curve=curve,
        aggregation=args.aggregation,
        mpc_horizon_hours=args.mpc_horizon,
        terminal_value_scale=args.terminal_value_scale,
    )

    print(f"Scenario     : {scenario.label()}")
    print(f"Sizing       : {sizing.solar_mw:.2f} MW-DC solar, {sizing.battery_mw:.2f} MW / "
          f"{sizing.battery_mwh:.1f} MWh battery")
    print(f"GPU curve    : {curve.provenance.name}  [{curve.provenance.kind}]  "
          f"aggregation={args.aggregation}")
    print(f"SOC boundary : cyclic (year must be self-sustaining)")
    print(f"MPC horizon  : {args.mpc_horizon} h, terminal value scale {args.terminal_value_scale}")
    print()

    runs = run_experiment_a(
        site, sizing, strategies=strategies,
        progress=lambda s: print(f"  running {s} ...", flush=True), **kwargs
    )
    _print_table(runs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"experiment_a_{scenario.label()}.json"
    out.write_text(json.dumps(
        {k: {"metrics": r.metrics, "metadata": r.metadata} for k, r in runs.items()},
        indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out}")

    fig = RESULTS_DIR / f"difficult_window_{scenario.label()}.png"
    start, end = plot_difficult_window(runs, site=site, path=fig, window_hours=args.window_hours)
    print(f"Wrote {fig}  (hours {start}-{end})")

    if args.derate_sweep:
        _sweep(site, sizing, strategies, kwargs, scenario)
    return 0


def _print_table(runs: dict) -> None:
    base = runs["fixed_load"].metrics["compute_units"]
    print("\n" + "=" * 96)
    print("EXPERIMENT A — same weather, same hardware, same demand; only the policy differs")
    print("=" * 96)
    header = (f"{'strategy':<26}{'compute':>10}{'vs fixed':>10}{'curtail %':>11}"
              f"{'unserved MWh':>14}{'thr h':>7}{'park h':>8}{'min SOC':>9}")
    print(header)
    print("-" * 96)
    for name, run in runs.items():
        m = run.metrics
        print(f"{name:<26}{m['compute_units']:>10.2f}"
              f"{100 * (m['compute_units'] / base - 1):>+9.3f}%"
              f"{m['solar_curtailed_pct']:>11.2f}"
              f"{m['involuntary_shortfall_mwh']:>14.2f}"
              f"{m['hours_throttled']:>7}{m['hours_parked']:>8}"
              f"{m['soc_min_mwh']:>9.1f}")

    annual = runs.get("perfect_foresight_annual")
    if annual and "planner_model_gap" in annual.metrics:
        gap = annual.metrics["planner_model_gap"]
        print(f"\nLP model vs simulator: predicted "
              f"{annual.metrics['planner_predicted_compute_units']:.4f}, "
              f"measured {annual.metrics['compute_units']:.4f}, gap {gap:+.6f}")
        if abs(gap) > 1e-6:
            print("  WARNING: the planner and the simulator disagree. Investigate before "
                  "trusting any MPC result.")


def _sweep(site, sizing, strategies, kwargs, scenario) -> None:  # noqa: C901
    print("\n" + "=" * 96)
    print("DE-RATED SWEEP — compute retained as the plant shrinks")
    print("=" * 96)
    sweep = derate_sweep(
        site, sizing, DEFAULT_SCALES, strategies=strategies,
        progress=lambda s: print(f"  {s} ...", flush=True), **kwargs
    )

    names = list(strategies)
    # Compute is never shown without unserved energy beside it: a strategy can
    # always buy compute by browning out, and the pair is the only honest read.
    print(f"\n{'scale':>6}{'solar':>8}{'batt':>7}" +
          "".join(f"{n[:20]:>21}" for n in names))
    print(f"{'':>6}{'MW':>8}{'MW':>7}" +
          "".join(f"{'compute / unserved':>21}" for _ in names))
    print("-" * (21 + 21 * len(names)))
    for scale in DEFAULT_SCALES:
        row = sweep[scale]
        s = sizing.scaled(scale)
        cells = "".join(
            f"{row[n].metrics['compute_units']:>13.1f} /{row[n].metrics['involuntary_shortfall_mwh']:>6.0f}"
            for n in names
        )
        print(f"{scale:>6.2f}{s.solar_mw:>8.1f}{s.battery_mw:>7.1f}{cells}")
    print("\n  unserved = MWh of GPU power requested but not delivered (a reliability")
    print("  failure, not a choice). Compute bought with brownouts is not comparable.")

    # The planner is a separate model of the plant; check it agreed with the
    # simulator at *every* sizing, not just the headline one.
    gaps = [
        (scale, row["perfect_foresight_annual"].metrics.get("planner_model_gap", 0.0))
        for scale, row in sweep.items()
        if "perfect_foresight_annual" in row
    ]
    worst = max((abs(g) for _, g in gaps), default=0.0)
    print(f"\nLP model vs simulator across the sweep: worst gap {worst:.3e} compute-units")
    if worst > 1e-6:
        for scale, gap in gaps:
            if abs(gap) > 1e-6:
                print(f"  WARNING scale {scale:.2f}: planner and simulator disagree by {gap:+.6f}")

    # The Experiment B question, read off an Experiment A sweep.
    print("\n" + "-" * 96)
    print("INFRASTRUCTURE NEEDED FOR EQUAL COMPUTE (interpolated from the sweep above)")
    print("-" * 96)
    print(f"{'compute target':>34}{'fixed scale':>14}" +
          "".join(f"{n[:16]:>18}" for n in names if n != "fixed_load"))
    for target_scale in (1.00, 0.60, 0.40):
        target = sweep[target_scale]["fixed_load"].metrics["compute_units"]
        cells = ""
        for n in names:
            if n == "fixed_load":
                continue
            found = infrastructure_for_target_compute(sweep, n, target)
            cells += (f"{found:>13.3f} ({100 * (found / target_scale - 1):+.1f}%)"
                      if found else f"{'n/a':>18}")
        print(f"{f'fixed_load @ {target_scale:.2f} = {target:.0f}':>34}"
              f"{target_scale:>14.2f}{cells}")
    print("\n  Read as: the scale factor each policy needs to match fixed_load's compute,")
    print("  and the implied change in solar+BESS capacity. Interpolated between sampled")
    print("  scales, so indicative of magnitude only — Experiment B optimises properly.")

    out = RESULTS_DIR / f"derate_sweep_{scenario.label()}.json"
    out.write_text(json.dumps(
        {str(scale): {n: r.metrics for n, r in row.items()} for scale, row in sweep.items()},
        indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out}")

    fig = RESULTS_DIR / f"derate_sweep_{scenario.label()}.png"
    curve_meta = next(iter(next(iter(sweep.values())).values())).metadata["gpu"]["curve"]
    plot_derate_sweep(sweep, site=site, path=fig, curve_meta=curve_meta)
    print(f"Wrote {fig}")


if __name__ == "__main__":
    raise SystemExit(main())
