#!/usr/bin/env python
"""How far ahead does a controller actually need to see?

Sweeps the MPC lookahead horizon against the annual perfect-foresight ceiling,
at both the fixed-load-optimal sizing and a de-rated one. The de-rated case is
the informative one: at the fixed-load optimum almost nothing binds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.experiments import Sizing, run_strategy          # noqa: E402
from flexcompute.scenario import LOCATIONS, Scenario              # noqa: E402
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot      # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--gpus", type=int, default=10_000)
    p.add_argument("--horizons", type=int, nargs="+", default=[12, 24, 48, 96, 168])
    p.add_argument("--scales", type=float, nargs="+", default=[1.00, 0.40])
    p.add_argument("--terminal-value-scales", type=float, nargs="+", default=[0.95])
    args = p.parse_args()

    logging.disable(logging.INFO)
    scenario = Scenario(location=args.location, total_gpus=args.gpus)
    site = scenario.build()
    s = load_snapshot(SNAPSHOT_DIR / f"{scenario.label()}.json")["optimized"]["sizing"]
    base = Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"])

    out: dict = {}
    for scale in args.scales:
        sizing = base.scaled(scale)
        fixed = run_strategy(site, "fixed_load", sizing).metrics["compute_units"]
        ceiling = run_strategy(site, "perfect_foresight_annual", sizing).metrics["compute_units"]
        print(f"\n=== scale {scale:.2f}  ({sizing.solar_mw:.1f} MW-DC / "
              f"{sizing.battery_mw:.1f} MW / {sizing.battery_mwh:.0f} MWh) ===")
        print(f"  {'fixed_load':<34}{fixed:>10.1f}")
        rows = {"fixed_load": fixed, "ceiling": ceiling, "mpc": {}}
        for tvs in args.terminal_value_scales:
            for horizon in args.horizons:
                started = time.time()
                m = run_strategy(
                    site, "perfect_foresight_mpc", sizing,
                    mpc_horizon_hours=horizon, terminal_value_scale=tvs,
                ).metrics
                label = f"MPC H={horizon}h tvs={tvs}"
                print(f"  {label:<34}{m['compute_units']:>10.1f}"
                      f"  ({100 * (m['compute_units'] / fixed - 1):+6.2f}% vs fixed,"
                      f" {100 * m['compute_units'] / ceiling:6.2f}% of ceiling,"
                      f" {m['involuntary_shortfall_mwh']:.1f} MWh unserved,"
                      f" {time.time() - started:.0f}s)", flush=True)
                rows["mpc"][label] = m["compute_units"]
        print(f"  {'annual LP (ceiling)':<34}{ceiling:>10.1f}")
        out[f"{scale:.2f}"] = rows

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"horizon_sweep_{scenario.label()}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
