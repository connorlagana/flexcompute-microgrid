#!/usr/bin/env python
"""Redraw the forecast-error figure from a saved sweep JSON.

Separate from the sweep itself so the figure can be restyled without paying
the ~40 minutes of MPC solves again. Pulls the perfect-foresight horizon curve
out of the horizon sweep if one is on disk, so the right panel can show what
lookahead was worth *before* the forecast could be wrong.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.plotting import plot_forecast_sweep   # noqa: E402
from flexcompute.scenario import LOCATIONS, Scenario   # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def perfect_horizon_curve(scenario: Scenario, scale: str) -> dict[str, float] | None:
    """The perfect-foresight MPC's compute by horizon, if the sweep exists."""
    path = RESULTS_DIR / f"horizon_sweep_{scenario.label()}.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text()).get(scale, {}).get("mpc", {})
    out: dict[str, float] = {}
    for label, compute in rows.items():
        # Labels look like "MPC H=48h tvs=0.95"
        try:
            hours = float(label.split("H=")[1].split("h")[0])
        except (IndexError, ValueError):
            continue
        out[str(hours)] = compute
    return {float(k): v for k, v in out.items()} or None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--gpus", type=int, default=10_000)
    p.add_argument("--scale", default="0.40")
    args = p.parse_args()

    logging.disable(logging.INFO)
    scenario = Scenario(location=args.location, total_gpus=args.gpus)
    source = RESULTS_DIR / f"forecast_sweep_{scenario.label()}.json"
    if not source.exists():
        raise SystemExit(f"No sweep at {source}. Run scripts/run_forecast_sweep.py.")

    data = json.loads(source.read_text())
    site = scenario.build()

    experiment = RESULTS_DIR / f"experiment_a_{scenario.label()}.json"
    curve_meta = None
    if experiment.exists():
        runs = json.loads(experiment.read_text()).get("runs", {})
        for run in runs.values():
            curve = run.get("metadata", {}).get("gpu", {}).get("curve")
            if curve:
                curve_meta = curve
                break

    path = RESULTS_DIR / f"forecast_sweep_{scenario.label()}.png"
    plot_forecast_sweep(
        data, site=site, path=path, scale=args.scale, curve_meta=curve_meta,
        perfect_horizon=perfect_horizon_curve(scenario, args.scale),
    )
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
