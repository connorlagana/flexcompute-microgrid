#!/usr/bin/env python
"""Render the five headline figures from the committed result files.

    python scripts/make_figures.py
    python scripts/make_figures.py --only 2 4
    python scripts/make_figures.py --drought-scale 0.25

Figures 1, 3, 4 and 5 are rendered from result JSON, so they are cheap and
reproducible without re-simulating. Figure 2 is a trace and needs hourly data,
so it re-runs four strategies on the single hardest solar drought found in the
whole 15-year record — the window is chosen from the *weather*, identically for
every strategy, never from which controller happened to struggle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from flexcompute.experiments import Sizing, run_strategy  # noqa: E402
from flexcompute.gpu import get_curve                     # noqa: E402
from flexcompute.multiyear import (                       # noqa: E402
    DALLAS_STUDY_YEARS,
    worst_solar_window,
)
from flexcompute.plotting import (                        # noqa: E402
    plot_controller_value_vs_scarcity,
    plot_difficult_window,
    plot_economic_comparison,
    plot_forecast_error_sensitivity,
    plot_year_distribution,
)
from flexcompute.scenario import Scenario                 # noqa: E402
from flexcompute.snapshot import SNAPSHOT_DIR, load_snapshot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

EXP_A = "experiment_a_multiyear_dallas_2010_2024_open_meteo_era5_cf0.00_h100_llama3_8b_pretrain_mayr2026.json"
SWEEP = "forecast_sweep_multiyear_dallas_2010_2024_open_meteo_era5_scale0.25.json"

#: Figure 2's cast, exactly as specified for the headline: no control, the best
#: forecast-free rule, the deployable forecast-aware design, and the ceiling.
DROUGHT_STRATEGIES = (
    "fixed_load", "casey_governor", "forecast_mpc", "perfect_foresight_annual",
)


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        print(f"  skipped: {path.name} not found")
        return None
    return json.loads(path.read_text())


def figure_1(data: dict) -> None:
    path = plot_controller_value_vs_scarcity(
        data["aggregates"], data["per_year"],
        path=RESULTS / "fig1_controller_value_vs_scarcity.png",
        n_years=data["year_set"]["n_years"],
    )
    print(f"  wrote {path.name}")


def figure_2(*, scale: float, window_hours: int) -> None:
    """Trace the hardest drought in the whole record."""
    print("  locating the hardest solar drought in the 15-year record ...")
    worst = None
    for year in DALLAS_STUDY_YEARS:
        site = Scenario(weather_year=year).build()
        drought = worst_solar_window(site, window_hours)
        if worst is None or drought.mean_solar_fraction < worst[1].mean_solar_fraction:
            worst = (year, drought, site)
    year, drought, site = worst
    print(f"    {year}, starting {drought.start_label}, "
          f"mean solar {drought.mean_solar_fraction:.4f} of nameplate "
          f"({drought.severity:.1%} of that year's average)")

    s = load_snapshot(SNAPSHOT_DIR / "dallas_10kgpu_ac_coupled_mv_coupled_pvgis.json")
    ref = s["optimized"]["sizing"]
    sizing = Sizing(ref["solar_mw_dc"], ref["battery_mw"], ref["battery_duration_h"]).scaled(scale)

    runs = {}
    for strategy in DROUGHT_STRATEGIES:
        started = time.time()
        runs[strategy] = run_strategy(site, strategy, sizing)
        print(f"    {strategy:<26} {time.time() - started:5.1f}s")

    # Center the drought in the frame with a margin either side, and keep the
    # window inside the year: the trace is positional and cannot wrap.
    margin = window_hours // 4
    start = int(np.clip(drought.start_hour - margin, 0, 8760 - window_hours - 2 * margin))
    curve = get_curve()
    plot_difficult_window(
        runs, site=site,
        path=RESULTS / "fig2_hardest_drought.png",
        window_hours=window_hours + 2 * margin,
        start_hour=start,
        title=(f"The hardest solar drought in 15 Dallas years — "
               f"{year}, from {drought.start_label}"),
        subtitle=(
            f"{sizing.solar_mw:.0f} MW-DC solar · {sizing.battery_mw:.0f} MW / "
            f"{sizing.battery_mwh:.0f} MWh battery (scale {scale:.2f}) · "
            f"{window_hours} h averaging {drought.mean_solar_fraction:.3f} of nameplate, "
            f"{drought.severity:.0%} of the year's mean · "
            f"GPU curve: {curve.provenance.name} [{curve.provenance.kind}]"
        ),
    )
    print(f"  wrote fig2_hardest_drought.png")


def figure_3(sweep: dict) -> None:
    rows = [
        {
            "nrmse": r["realised_nrmse"]["median"],
            "median": r["compute"]["median"],
            "p10": r["compute"]["p10"],
            "p90": r["compute"]["p90"],
        }
        for r in sweep["rows"]
    ]
    targets = [str(r["target_nrmse_pct"]) for r in sweep["rows"]]
    series = [
        [year["forecast_mpc"][t]["compute_units"] for t in targets]
        for year in sweep["per_year"].values()
    ]
    path = plot_forecast_error_sensitivity(
        rows,
        path=RESULTS / "fig3_forecast_error_sensitivity.png",
        fixed_load_compute=sweep["references"]["fixed_load"]["median"],
        ceiling_compute=sweep["references"]["perfect_foresight_annual"]["median"],
        scale=sweep["infrastructure_scale"],
        n_years=sweep["year_set"]["n_years"],
        per_year=series,
    )
    print(f"  wrote {path.name}")


def figure_4(economics: dict, *, variant: str) -> None:
    designs = []
    for label, r in economics["results"].items():
        v, mode, strategy = label.split("|")
        if v != variant or mode != "free":
            continue
        designs.append({
            "strategy": strategy,
            "capex_musd": r["capex_musd"],
            "solar_mw": r["sizing"]["solar_mw_dc"],
            "battery_mw": r["sizing"]["battery_mw"],
            "battery_mwh": r["sizing"]["battery_mwh"],
            "duration_h": r["sizing"]["battery_duration_h"],
            "duration_extrapolated": r["duration_extrapolated"],
        })
    if not designs:
        print(f"  skipped: no {variant} free-duration results")
        return

    order = {"fixed_load": 0, "casey_governor": 1, "forecast_mpc": 2,
             "perfect_foresight_annual": 3}
    designs.sort(key=lambda d: order.get(d["strategy"], 9))

    label = ("equal compute only (B1)" if variant == "B1" else
             f"equal compute AND shortfall ≤ {economics['reliability_cap_mwh']:,.0f} MWh/yr (B2)")
    path = plot_economic_comparison(
        designs,
        path=RESULTS / f"fig4_economics_{variant.lower()}.png",
        reference_capex=economics["reference_capex_musd"],
        variant_label=label,
    )
    print(f"  wrote {path.name}")


def figure_5(data: dict, *, scale: float) -> None:
    path = plot_year_distribution(
        data["per_year"],
        path=RESULTS / "fig5_year_distribution.png",
        scale=scale,
        concentration=data["concentration"].get(str(scale)),
    )
    print(f"  wrote {path.name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", type=int, choices=[1, 2, 3, 4, 5], default=None)
    p.add_argument("--drought-scale", type=float, default=0.25)
    p.add_argument("--drought-hours", type=int, default=72)
    p.add_argument("--distribution-scale", type=float, default=0.25)
    p.add_argument("--economics", default=None,
                   help="path to an experiment_b2_*.json")
    args = p.parse_args()

    import logging
    logging.disable(logging.INFO)
    wanted = set(args.only or [1, 2, 3, 4, 5])
    RESULTS.mkdir(parents=True, exist_ok=True)

    if 1 in wanted or 5 in wanted:
        data = _load(EXP_A)
        if data:
            if 1 in wanted:
                print("Figure 1 — controller value vs infrastructure scarcity")
                figure_1(data)
            if 5 in wanted:
                print("Figure 5 — distribution across real weather years")
                figure_5(data, scale=args.distribution_scale)

    if 2 in wanted:
        print("Figure 2 — the hardest multi-day weather event")
        figure_2(scale=args.drought_scale, window_hours=args.drought_hours)

    if 3 in wanted:
        print("Figure 3 — forecast-error sensitivity")
        sweep = _load(SWEEP)
        if sweep:
            figure_3(sweep)

    if 4 in wanted:
        print("Figure 4 — economic comparison")
        if args.economics:
            economics = json.loads(Path(args.economics).read_text())
        else:
            candidates = sorted(RESULTS.glob("experiment_b2_*.json"))
            economics = json.loads(candidates[-1].read_text()) if candidates else None
            if economics is None:
                print("  skipped: no experiment_b2_*.json found")
        if economics:
            for variant in ("B1", "B2"):
                figure_4(economics, variant=variant)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
