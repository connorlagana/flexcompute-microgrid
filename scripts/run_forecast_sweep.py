#!/usr/bin/env python
"""What does the forecast-aware controller lose by not knowing the future?

Every MPC number in this project so far was computed with perfect foresight,
which makes them a ceiling rather than a design. This sweep replaces the truth
with a belief -- same solver, same physics, same plant -- and measures what is
left.

Three questions, in order of importance:

1. **How much of the perfect-foresight advantage survives forecast error?**
   Swept over error level at the sizing where stored energy actually binds.
2. **Is the answer an artefact of one lucky error realisation?** Repeated over
   seeds at the reference error level.
3. **Does lookahead still help once the lookahead is wrong?** The horizon result
   in the README was measured with a *perfect* 24-hour forecast; a longer
   horizon now buys more foresight and more error at the same time.

    python scripts/run_forecast_sweep.py              # the full sweep (~1 h)
    python scripts/run_forecast_sweep.py --quick      # one error level, one scale
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
from flexcompute.forecast import NoisySolarForecast               # noqa: E402
from flexcompute.scenario import LOCATIONS, Scenario              # noqa: E402
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot      # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def noisy(sigma_24h: float, seed: int):
    """A forecast factory pinned to one error level and one realisation."""
    return lambda truth: NoisySolarForecast(truth, sigma_24h=sigma_24h, seed=seed)


def record(run, **extra) -> dict:
    m = run.metrics
    return {
        "compute_units": m["compute_units"],
        "involuntary_shortfall_mwh": m["involuntary_shortfall_mwh"],
        "hours_throttled": m["hours_throttled"],
        "hours_with_shortfall": m["hours_with_shortfall"],
        "solar_curtailed_pct": m["solar_curtailed_pct"],
        "cyclic_soc_residual_mwh": m.get("cyclic_soc_residual_mwh"),
        "cyclic_evaluations": run.metadata.get("cyclic_soc", {}).get("evaluations"),
        "forecast_nrmse_24h_pct_of_capacity": m.get(
            "forecast_nrmse_24h_pct_of_capacity"
        ),
        "wall_time_s": m["wall_time_s"],
        **extra,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--gpus", type=int, default=10_000)
    p.add_argument("--scales", type=float, nargs="+", default=[0.40, 1.00])
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.30])
    p.add_argument("--reference-sigma", type=float, default=0.15,
                   help="Error level used for the seed and horizon sweeps.")
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--extra-seeds", type=int, nargs="*", default=[1, 2],
                   help="Additional error realisations at the reference sigma.")
    p.add_argument("--horizons", type=int, nargs="*", default=[12, 24, 96],
                   help="Extra lookaheads to test at the reference sigma (48 always runs).")
    p.add_argument("--horizon-scale", type=float, default=0.40,
                   help="Sizing at which to sweep the horizon.")
    p.add_argument("--mpc-horizon", type=int, default=48)
    p.add_argument("--quick", action="store_true",
                   help="One scale, one error level, no seed or horizon sweep.")
    args = p.parse_args()

    if args.quick:
        args.scales = [0.40]
        args.sigmas = [args.reference_sigma]
        args.extra_seeds = []
        args.horizons = []

    logging.disable(logging.INFO)
    scenario = Scenario(location=args.location, total_gpus=args.gpus)
    site = scenario.build()
    s = load_snapshot(SNAPSHOT_DIR / f"{scenario.label()}.json")["optimized"]["sizing"]
    base = Sizing(s["solar_mw_dc"], s["battery_mw"], s["battery_duration_h"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"forecast_sweep_{scenario.label()}.json"
    out: dict = {
        "scenario": scenario.label(),
        "base_sizing": base.as_dict(),
        "mpc_horizon_hours": args.mpc_horizon,
        "reference_sigma_24h": args.reference_sigma,
        "seed": args.seed,
        "scales": {},
        "horizon_sweep": {},
    }

    def checkpoint() -> None:
        """Write after every run: an hour-long sweep must survive interruption."""
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    def run(label: str, *a, **kw):
        started = time.time()
        result = run_strategy(site, *a, **kw)
        m = result.metrics
        print(f"  {label:<38}{m['compute_units']:>9.1f}"
              f"  {m['involuntary_shortfall_mwh']:>8.1f} MWh unserved"
              f"  {time.time() - started:>5.0f}s", flush=True)
        return result

    for scale in args.scales:
        sizing = base.scaled(scale)
        print(f"\n=== scale {scale:.2f}  ({sizing.solar_mw:.1f} MW-DC / "
              f"{sizing.battery_mw:.1f} MW / {sizing.battery_mwh:.0f} MWh) ===", flush=True)
        rows: dict = {"sizing": sizing.as_dict(), "runs": {}}
        out["scales"][f"{scale:.2f}"] = rows

        rows["runs"]["fixed_load"] = record(run("fixed_load", "fixed_load", sizing))
        checkpoint()
        rows["runs"]["perfect_foresight_annual"] = record(
            run("perfect_foresight_annual (ceiling)", "perfect_foresight_annual", sizing)
        )
        checkpoint()
        rows["runs"]["perfect_foresight_mpc"] = record(
            run(f"perfect_foresight_mpc H={args.mpc_horizon}h",
                "perfect_foresight_mpc", sizing, mpc_horizon_hours=args.mpc_horizon),
            horizon_h=args.mpc_horizon,
        )
        checkpoint()

        for sigma in args.sigmas:
            key = f"forecast_mpc_sigma{sigma:.2f}_seed{args.seed}"
            rows["runs"][key] = record(
                run(f"forecast_mpc sigma={sigma:.2f} seed={args.seed}",
                    "forecast_mpc", sizing, mpc_horizon_hours=args.mpc_horizon,
                    forecast_factory=noisy(sigma, args.seed)),
                sigma_24h=sigma, seed=args.seed, horizon_h=args.mpc_horizon,
            )
            checkpoint()

        # Is the answer a property of the error level, or of one draw from it?
        for seed in args.extra_seeds:
            key = f"forecast_mpc_sigma{args.reference_sigma:.2f}_seed{seed}"
            rows["runs"][key] = record(
                run(f"forecast_mpc sigma={args.reference_sigma:.2f} seed={seed}",
                    "forecast_mpc", sizing, mpc_horizon_hours=args.mpc_horizon,
                    forecast_factory=noisy(args.reference_sigma, seed)),
                sigma_24h=args.reference_sigma, seed=seed, horizon_h=args.mpc_horizon,
            )
            checkpoint()

    # -- does lookahead still pay when the lookahead is wrong? ---------------
    if args.horizons:
        sizing = base.scaled(args.horizon_scale)
        print(f"\n=== horizon sweep at scale {args.horizon_scale:.2f}, "
              f"sigma={args.reference_sigma:.2f} ===", flush=True)
        swept = out["horizon_sweep"]
        swept["scale"] = args.horizon_scale
        swept["sigma_24h"] = args.reference_sigma
        swept["runs"] = {}
        for horizon in args.horizons:
            swept["runs"][f"H{horizon}"] = record(
                run(f"forecast_mpc H={horizon}h", "forecast_mpc", sizing,
                    mpc_horizon_hours=horizon,
                    forecast_factory=noisy(args.reference_sigma, args.seed)),
                sigma_24h=args.reference_sigma, seed=args.seed, horizon_h=horizon,
            )
            checkpoint()

    checkpoint()
    summarise(out)
    print(f"\nWrote {path}")
    return 0


def summarise(out: dict) -> None:
    """The one number the sweep exists to produce, per scale.

    *Retention* is the share of the perfect-foresight advantage that survives
    not knowing the future:

        (compute_forecast - compute_fixed) / (compute_ceiling - compute_fixed)

    Reported against the annual LP rather than the receding-horizon MPC, so it
    charges forecast error for the cost of a finite horizon too -- the harsher
    and more honest denominator. Unserved energy is printed beside it because a
    controller can always buy compute by browning out (ASSUMPTIONS B11).
    """
    for scale, block in sorted(out["scales"].items()):
        runs = block["runs"]
        fixed = runs["fixed_load"]["compute_units"]
        ceiling = runs["perfect_foresight_annual"]["compute_units"]
        window = ceiling - fixed
        print(f"\n--- scale {scale}: perfect foresight is worth "
              f"{window:+.1f} compute-units over no control ---")
        print(f"  {'forecast':<34}{'nRMSE@24h':>10}{'compute':>10}"
              f"{'retained':>10}{'unserved':>11}")
        for name, row in sorted(runs.items(), key=lambda kv: (
                kv[1].get("sigma_24h", -1.0), kv[1].get("seed", 0))):
            if window <= 0:
                retained = "n/a"
            else:
                retained = f"{100 * (row['compute_units'] - fixed) / window:.1f}%"
            nrmse = row.get("forecast_nrmse_24h_pct_of_capacity")
            print(f"  {name:<34}{(f'{nrmse:.1f}%' if nrmse else '—'):>10}"
                  f"{row['compute_units']:>10.1f}{retained:>10}"
                  f"{row['involuntary_shortfall_mwh']:>10.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
