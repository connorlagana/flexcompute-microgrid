"""Baseline snapshots: the reproducibility contract.

A snapshot is a JSON record of everything a scenario produces -- provenance,
dispatch metrics at several fixed sizings, and the optimised sizing with its
LCOE. ``scripts/run_baseline.py --check`` re-runs the model and fails on any
numeric drift.

The point is *not* to prove the reference model is correct. It is to make any
change in its output visible and deliberate. The core invariant of the whole
project is that introducing a controller abstraction must leave the fixed-load
numbers bit-identical; without a snapshot, a silent regression is undetectable.

Volatile fields (wall-clock timings, fetch timestamps) are excluded from
comparison. Everything else, including the upstream git SHA, is compared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .baseline import evaluate_fixed_sizing, optimize_sizing
from .scenario import Site
from .upstream_bridge import PROJECT_ROOT

SNAPSHOT_DIR = PROJECT_ROOT / "baselines"

#: Fixed sizings to probe, as multiples of the facility *design* load (MW).
#: Chosen to straddle the interesting regimes: clearly infeasible, marginal
#: against the 99% uptime constraint, and comfortably over-built.
PROBE_MULTIPLES: tuple[tuple[float, float], ...] = (
    (10.0, 3.0),
    (15.0, 7.0),
    (21.0, 10.0),
    (28.0, 14.0),
)

#: Keys whose values change run-to-run and carry no modelling information.
VOLATILE_KEYS = frozenset({"retrieved_utc", "wall_time_s"})

DEFAULT_RTOL = 1e-9


def build_snapshot(site: Site, *, run_optimizer: bool = True) -> dict:
    """Run a scenario end to end and return its snapshot dict."""
    design_mw = float(site.facility_load.facility_load_design_mw)

    probes = []
    for solar_mult, battery_mult in PROBE_MULTIPLES:
        solar_mw = round(solar_mult * design_mw, 6)
        battery_mw = round(battery_mult * design_mw, 6)
        run = evaluate_fixed_sizing(site, solar_mw, battery_mw)
        run.assert_physical()
        probes.append(
            {
                "solar_multiple_of_design": solar_mult,
                "battery_multiple_of_design": battery_mult,
                "metrics": run.metrics,
                "audit": run.audit.as_dict(),
            }
        )

    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "provenance": site.provenance(),
        "design_load_mw": design_mw,
        "fixed_sizing_probes": probes,
    }
    if run_optimizer:
        snapshot["optimized"] = optimize_sizing(site)
    return snapshot


def write_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")


def load_snapshot(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _walk(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in VOLATILE_KEYS:
                continue
            yield from _walk(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from _walk(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare_snapshots(expected: dict, actual: dict, rtol: float = DEFAULT_RTOL) -> list[str]:
    """Return human-readable descriptions of every difference found."""
    exp = dict(_walk(expected))
    act = dict(_walk(actual))
    diffs: list[str] = []

    for key in sorted(set(exp) - set(act)):
        diffs.append(f"{key}: missing from new run (was {exp[key]!r})")
    for key in sorted(set(act) - set(exp)):
        diffs.append(f"{key}: new key in this run ({act[key]!r})")

    for key in sorted(set(exp) & set(act)):
        a, b = exp[key], act[key]
        if isinstance(a, bool) or isinstance(b, bool):
            if a is not b:
                diffs.append(f"{key}: {a!r} -> {b!r}")
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            scale = max(abs(a), abs(b))
            if abs(a - b) > rtol * scale:
                diffs.append(f"{key}: {a!r} -> {b!r}  (rel {abs(a - b) / scale:.3e})")
        elif a != b:
            diffs.append(f"{key}: {a!r} -> {b!r}")
    return diffs
