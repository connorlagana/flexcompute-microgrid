#!/usr/bin/env python
"""Produce (or refresh) the fixed-load baseline snapshot for a scenario.

    python scripts/run_baseline.py                     # dallas, 10k GPUs, PVGIS
    python scripts/run_baseline.py --location phoenix
    python scripts/run_baseline.py --weather-source nsrdb    # needs NLR_API_KEY
    python scripts/run_baseline.py --check                   # verify, do not write

The snapshot is the reproducibility contract: ``--check`` re-runs the model and
fails if any recorded number moved. Run it after any change that is supposed to
leave fixed-load behaviour untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.baseline import evaluate_fixed_sizing, optimize_sizing  # noqa: E402
from flexcompute.scenario import LOCATIONS, Scenario                     # noqa: E402
from flexcompute.snapshot import (                                       # noqa: E402
    SNAPSHOT_DIR,
    build_snapshot,
    compare_snapshots,
    load_snapshot,
    write_snapshot,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--gpus", type=int, default=10_000)
    p.add_argument("--uptime", type=float, default=99.0)
    p.add_argument("--architecture", default="ac_coupled", choices=["ac_coupled", "dc_coupled"])
    p.add_argument("--topology", default="mv_coupled", choices=["mv_coupled", "lv_direct"])
    p.add_argument("--weather-source", default="pvgis", choices=["pvgis", "nsrdb"])
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--skip-optimizer", action="store_true",
                   help="Only run the fixed-sizing probes (seconds instead of minutes).")
    p.add_argument("--check", action="store_true",
                   help="Compare against the stored snapshot instead of writing it.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.disable(logging.INFO)

    scenario = Scenario(
        location=args.location,
        total_gpus=args.gpus,
        required_uptime_pct=args.uptime,
        architecture=args.architecture,
        topology=args.topology,
        weather_source=args.weather_source,
        seed=args.seed,
    )

    print(f"Scenario: {scenario.label()}  (key {scenario.key()})")
    print(f"  {json.dumps(asdict(scenario))}")
    print("Building site ...", flush=True)
    site = scenario.build()
    print(f"  cooling case {site.cooling_case}, annual PUE {site.annual_pue:.4f}, "
          f"IT avg {site.it_load_avg_mw:.3f} MW, design {site.facility_load.facility_load_design_mw:.3f} MW")

    if not args.skip_optimizer:
        print("Optimising sizing (this takes a few minutes) ...", flush=True)
    snapshot = build_snapshot(site, run_optimizer=not args.skip_optimizer)

    path = SNAPSHOT_DIR / f"{scenario.label()}.json"

    if args.check:
        if not path.exists():
            print(f"FAIL: no stored snapshot at {path}")
            return 1
        diffs = compare_snapshots(load_snapshot(path), snapshot)
        if diffs:
            print(f"FAIL: {len(diffs)} value(s) changed vs {path.name}")
            for d in diffs:
                print(f"  {d}")
            return 1
        print(f"OK: reproduces {path.name} exactly")
        return 0

    # A --skip-optimizer run produces a strictly smaller snapshot. Writing it
    # over a complete one would silently delete the optimised baseline, which
    # is the number the whole project is measured against.
    if args.skip_optimizer and path.exists() and "optimized" in load_snapshot(path):
        print(
            f"\nRefusing to overwrite {path.name}: it contains an optimised "
            "baseline and this run skipped the optimizer.\n"
            "Re-run without --skip-optimizer, or delete the file deliberately."
        )
        _report(snapshot)
        return 1

    write_snapshot(path, snapshot)
    print(f"\nWrote {path}")
    _report(snapshot)
    return 0


def _report(snapshot: dict) -> None:
    probes = snapshot["fixed_sizing_probes"]
    print("\nFixed-sizing probes (fixed load, no controller)")
    print(f"  {'solar MW':>9} {'batt MW':>8} {'batt MWh':>9} {'uptime %':>9} "
          f"{'served %':>9} {'curtail %':>10} {'cycles/y':>9}")
    for probe in probes:
        m = probe["metrics"]
        print(f"  {m['solar_mw_dc']:9.1f} {m['battery_mw']:8.1f} {m['battery_mwh']:9.1f} "
              f"{m['uptime_pct']:9.3f} {m['energy_served_pct']:9.3f} "
              f"{m['solar_curtailed_pct']:10.2f} {m['battery_cycles_per_year']:9.1f}")

    opt = snapshot.get("optimized")
    if not opt:
        return
    s, c, y = opt["sizing"], opt["cost"], opt["year_0"]
    print("\nOptimised fixed-load baseline (upstream sizing search)")
    print(f"  solar                {s['solar_mw_dc']:12.2f} MW-DC")
    print(f"  battery power        {s['battery_mw']:12.2f} MW")
    print(f"  battery energy       {s['battery_mwh']:12.2f} MWh  ({s['battery_duration_h']:.0f} h)")
    print(f"  land                 {s['land_area_acres']:12.1f} acres")
    print(f"  uptime (year 0)      {y['uptime_pct']:12.3f} %")
    print(f"  energy served        {y['energy_served_pct']:12.3f} %")
    print(f"  load served          {y['load_served_mwh']:12.1f} MWh/yr")
    print(f"  unmet load           {y['unmet_load_mwh']:12.1f} MWh/yr")
    print(f"  solar generation     {y['solar_generation_mwh']:12.1f} MWh/yr")
    print(f"  solar curtailed      {y['solar_curtailed_mwh']:12.1f} MWh/yr "
          f"({y['solar_curtailed_pct']:.1f} %)")
    print(f"  battery throughput   {y['battery_discharged_mwh']:12.1f} MWh/yr discharged")
    print(f"  battery cycles       {y['battery_cycles_per_year']:12.1f} /yr")
    print(f"  CAPEX (opt objective){c['capex_optimizer_objective_musd']:12.1f} M$ (2022)")
    print(f"  CAPEX NPV            {c['capex_npv_usd']/1e6:12.1f} M$")
    print(f"  LCOE                 {c['lcoe_usd_per_kwh']:12.4f} $/kWh")
    print(f"  optimizer            {opt['optimizer']['function_evaluations']} evals, "
          f"{opt['optimizer']['wall_time_s']} s")


if __name__ == "__main__":
    raise SystemExit(main())
