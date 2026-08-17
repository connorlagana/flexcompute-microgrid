#!/usr/bin/env python
"""Redraw the de-rated sweep figure from a saved sweep JSON.

Separate from the sweep itself so the figure can be restyled without paying
the ~30 minutes of MPC solves again.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcompute.plotting import plot_derate_sweep   # noqa: E402
from flexcompute.scenario import LOCATIONS, Scenario  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@dataclass
class _Row:
    """Minimal stand-in for a DispatchResult: the plot only reads metrics."""
    metrics: dict


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--location", default="dallas", choices=sorted(LOCATIONS))
    p.add_argument("--gpus", type=int, default=10_000)
    args = p.parse_args()

    logging.disable(logging.INFO)
    scenario = Scenario(location=args.location, total_gpus=args.gpus)
    source = RESULTS_DIR / f"derate_sweep_{scenario.label()}.json"
    if not source.exists():
        raise SystemExit(f"No sweep at {source}. Run run_experiment_a.py --derate-sweep.")

    raw = json.loads(source.read_text())
    sweep = {
        float(scale): {name: _Row(metrics) for name, metrics in row.items()}
        for scale, row in raw.items()
    }

    experiment = RESULTS_DIR / f"experiment_a_{scenario.label()}.json"
    curve_meta = None
    if experiment.exists():
        payload = json.loads(experiment.read_text())
        curve_meta = next(iter(payload.values()))["metadata"]["gpu"]["curve"]

    out = RESULTS_DIR / f"derate_sweep_{scenario.label()}.png"
    plot_derate_sweep(sweep, site=scenario.build(), path=out, curve_meta=curve_meta)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
