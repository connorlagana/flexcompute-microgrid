#!/usr/bin/env python
"""Experiment B: equal useful compute, minimum capital.

Every strategy must deliver the *same* annual compute as the reference
fixed-load plant. The question is how little solar and storage each one needs to
do it, with battery power and energy sized independently.

    python scripts/run_experiment_b.py
    python scripts/run_experiment_b.py --cooling-fixed-fraction 0.3
    python scripts/run_experiment_b.py --strategies fixed_load perfect_foresight_annual

The result is deliberately decomposed into two effects, because conflating them
would overstate the case for flexible control:

  1. changing the *metric* from 99% uptime to delivered compute, measured by
     re-sizing the fixed load itself;
  2. flexible operation on top of that.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.costs import from_upstream_config, upstream_equivalent_capex_usd  # noqa: E402
from flexcompute.experiments import (          # noqa: E402
    Sizing,
    minimum_capex_for_compute,
    run_strategy,
)
from flexcompute.gpu import get_curve          # noqa: E402
from flexcompute.scenario import LOCATIONS, Scenario  # noqa: E402
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_STRATEGIES = ("fixed_load", "simple_throttle", "perfect_foresight_annual")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--gpus", type=int, default=10_000)
    p.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    p.add_argument("--cooling-fixed-fraction", type=float, default=0.0)
    p.add_argument("--aggregation", default="time_shared", choices=["time_shared", "per_device"])
    p.add_argument("--curve", default=None)
    p.add_argument("--maxiter", type=int, default=20)
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260815)
    args = p.parse_args()

    logging.disable(logging.INFO)
    scenario = Scenario(location=args.location, total_gpus=args.gpus)
    site = scenario.build()
    snapshot = load_snapshot(SNAPSHOT_DIR / f"{scenario.label()}.json")
    s = snapshot["optimized"]["sizing"]
    reference = Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"])

    cost_model = from_upstream_config(site.config, scenario.architecture)
    curve = get_curve(args.curve) if args.curve else get_curve()
    run_kwargs = dict(
        curve=curve,
        aggregation=args.aggregation,
        cooling_fixed_fraction=args.cooling_fixed_fraction,
    )

    # Sanity: the decomposed cost model must reproduce upstream's capex at 4 h.
    ours = cost_model.total_capex_usd(reference.solar_mw, reference.battery_mw,
                                      reference.battery_mwh) / 1e6
    theirs = upstream_equivalent_capex_usd(site.config, reference.solar_mw,
                                           reference.battery_mw, scenario.architecture) / 1e6
    print(f"Scenario        : {scenario.label()}")
    print(f"Reference plant : {reference.solar_mw:.1f} MW-DC solar, "
          f"{reference.battery_mw:.1f} MW / {reference.battery_mwh:.0f} MWh "
          f"({reference.duration_h:.0f} h)")
    print(f"Cost model      : {cost_model.storage_cost_per_kw:.1f} $/kW + "
          f"{cost_model.storage_cost_per_kwh:.1f} $/kWh storage, "
          f"{cost_model.solar_cost_per_kw:.1f} $/kW solar")
    print(f"  reconciliation: ours {ours:.4f} M$ vs upstream {theirs:.4f} M$ "
          f"(diff {ours - theirs:+.2e})")
    if abs(ours - theirs) > 1e-6:
        print("  WARNING: cost decomposition does not reproduce upstream at 4 h.")
    print(f"GPU curve       : {curve.provenance.name} [{curve.provenance.kind}], "
          f"aggregation={args.aggregation}")
    print(f"Cooling fixed   : {args.cooling_fixed_fraction:.2f}")
    print()

    # The target: what the reference plant actually delivers under a fixed load.
    baseline = run_strategy(site, "fixed_load", reference, **run_kwargs)
    target = baseline.metrics["compute_units"]
    print(f"Compute target  : {target:,.2f} compute-unit-hours "
          f"(the reference plant's own fixed-load output)")
    print(f"Reference capex : {ours:,.1f} M$ (2022)")
    print()

    results = {}
    for strategy in args.strategies:
        print(f"  optimising {strategy} ...", flush=True)
        results[strategy] = minimum_capex_for_compute(
            site, strategy, target_compute=target, cost_model=cost_model,
            reference=reference, seed=args.seed, maxiter=args.maxiter,
            popsize=args.popsize, **run_kwargs,
        )
        r = results[strategy]
        print(f"     -> {r.capex_musd:,.1f} M$  "
              f"({r.sizing.solar_mw:.1f} MW / {r.sizing.battery_mw:.1f} MW / "
              f"{r.sizing.battery_mwh:.0f} MWh, {r.sizing.duration_h:.1f} h)  "
              f"compute {r.compute_units:,.1f}  {r.evaluations} evals, {r.wall_time_s:.0f}s")

    _report(results, reference_capex=ours, target=target)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / (
        f"experiment_b_{scenario.label()}_cf{args.cooling_fixed_fraction:.2f}.json"
    )
    out.write_text(json.dumps({
        "target_compute": target,
        "reference_sizing": reference.as_dict(),
        "reference_capex_musd": ours,
        "cost_model": cost_model.metadata(),
        "settings": {
            "cooling_fixed_fraction": args.cooling_fixed_fraction,
            "aggregation": args.aggregation,
            "curve": curve.provenance.name,
            "seed": args.seed,
        },
        "results": {k: v.as_dict() for k, v in results.items()},
    }, indent=2, sort_keys=True, default=float) + "\n")
    print(f"\nWrote {out}")
    return 0


def _report(results: dict, *, reference_capex: float, target: float) -> None:
    print("\n" + "=" * 96)
    print("EXPERIMENT B — minimum capital for equal annual compute")
    print("=" * 96)
    print(f"{'design':<28}{'CAPEX M$':>11}{'vs ref':>9}{'solar MW':>10}"
          f"{'batt MW':>9}{'batt MWh':>10}{'dur h':>7}{'unserved':>10}")
    print("-" * 96)
    print(f"{'reference (99% uptime)':<28}{reference_capex:>11.1f}{'—':>9}"
          f"{'':>10}{'':>9}{'':>10}{'':>7}{'':>10}")
    for name, r in results.items():
        print(f"{name:<28}{r.capex_musd:>11.1f}"
              f"{100 * (r.capex_musd / reference_capex - 1):>+8.1f}%"
              f"{r.sizing.solar_mw:>10.1f}{r.sizing.battery_mw:>9.1f}"
              f"{r.sizing.battery_mwh:>10.0f}{r.sizing.duration_h:>7.1f}"
              f"{r.involuntary_shortfall_mwh:>10.1f}")
        if not r.success:
            print(f"{'':28}  WARNING: missed the compute target "
                  f"({r.compute_units:,.1f} < {target:,.1f})")

    # The decomposition that keeps the claim honest.
    fixed = results.get("fixed_load")
    if fixed:
        print("\n" + "-" * 96)
        print("WHERE THE SAVING COMES FROM")
        print("-" * 96)
        metric_effect = 100 * (fixed.capex_musd / reference_capex - 1)
        print(f"  changing the metric (uptime -> compute), fixed load re-sized: "
              f"{metric_effect:+.1f}%")
        for name, r in results.items():
            if name == "fixed_load":
                continue
            extra = 100 * (r.capex_musd / fixed.capex_musd - 1)
            total = 100 * (r.capex_musd / reference_capex - 1)
            print(f"  flexible operation on top of that ({name}): {extra:+.1f}%"
                  f"   → {total:+.1f}% vs the reference plant")
        print("\n  Most of the first number is not about control at all: it is what happens")
        print("  when a plant stops being sized for a 99%-uptime tail. Attributing it to")
        print("  flexible operation would overstate the case substantially.")


if __name__ == "__main__":
    raise SystemExit(main())
