"""Figures.

One chart type so far: the "difficult window" trace, which is the visual test
of whether the controller architecture does anything. It shows, over the
hardest few days of the year:

* what the sun did (the driver, identical for every controller);
* the four power quantities per controller, arranged so that both gaps are
  *visible as gaps* -- demand-to-target is voluntary throttling, target-to-
  delivered is involuntary shortfall;
* what happened to the battery under each policy.

Design notes: no dual axes anywhere (each panel carries one measure, one
scale); controllers get a fixed categorical hue each, validated for
colour-vision deficiency; the unconstrained demand reference is drawn in
neutral ink so it never competes with controller identity.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

# Categorical slots from the reference palette, validated as a set on the light
# surface: worst adjacent CVD ΔE 9.2 (deutan), normal-vision ΔE 27.6, all inside
# the lightness band. Aqua sits below 3:1 contrast, so the relief rule applies --
# every SOC line carries a visible direct label. Do not substitute without
# re-running scripts/validate_palette.js.
CONTROLLER_COLORS = {
    "fixed_load": "#2a78d6",                # blue
    "simple_throttle": "#eb6834",           # orange
    "perfect_foresight_mpc": "#1baf7a",     # aqua
    "perfect_foresight_annual": "#4a3aa7",  # violet
}
FALLBACK_COLORS = ["#e87ba4", "#008300", "#e34948", "#eda100"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8c8b86"
GRID = "#e6e5e1"
SOLAR_FILL = "#eda100"
SOLAR_LINE = "#a87200"

REFERENCE_YEAR = 2023

#: Logical reporting order: no control -> naive -> optimal.
STRATEGY_DISPLAY_ORDER = (
    "fixed_load",
    "simple_throttle",
    "perfect_foresight_mpc",
    "perfect_foresight_annual",
)


def _color(name: str, index: int) -> str:
    return CONTROLLER_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def find_difficult_window(
    runs: Mapping[str, object], window_hours: int = 72, key: str = "involuntary_shortfall_mw"
) -> tuple[int, int]:
    """Locate the window where controllers are under the most stress.

    Scored on the *worst* controller at each hour, so the window is chosen by
    difficulty rather than by which policy happens to be on screen. Falls back
    to the lowest-solar window when nothing is ever short -- there is still a
    hardest few days even in a system that never fails.
    """
    series = [np.asarray(run.hourly[key].to_numpy(dtype=float)) for run in runs.values()]  # type: ignore[attr-defined]
    stress = np.max(np.vstack(series), axis=0)

    if stress.sum() <= 0:
        any_run = next(iter(runs.values()))
        stress = -np.asarray(any_run.hourly["solar_dc_mw"].to_numpy(dtype=float))  # type: ignore[attr-defined]
        stress = stress - stress.min()

    kernel = np.ones(window_hours)
    rolling = np.convolve(stress, kernel, mode="valid")
    start = int(np.argmax(rolling))
    return start, start + window_hours


def plot_difficult_window(
    runs: Mapping[str, object],
    *,
    site,
    path: Path,
    window_hours: int = 72,
) -> tuple[int, int]:
    """Render the difficult-window figure. Returns the (start, end) hours."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    start, end = find_difficult_window(runs, window_hours)
    sl = slice(start, end)
    names = list(runs)
    hours = np.arange(start, end)
    x = np.array(
        [
            datetime.datetime(REFERENCE_YEAR, 1, 1) + datetime.timedelta(hours=int(h))
            for h in hours
        ]
    )

    n_rows = 2 + len(names)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(12.5, 1.4 + 2.35 * n_rows), sharex=True,
        gridspec_kw={"hspace": 0.34, "top": 0.885, "bottom": 0.055,
                     "left": 0.062, "right": 0.985},
    )
    fig.patch.set_facecolor(SURFACE)

    first = next(iter(runs.values()))
    solar = first.hourly["solar_dc_mw"].to_numpy(dtype=float)[sl]  # type: ignore[attr-defined]

    # -- row 1: the driver -------------------------------------------------
    ax = axes[0]
    ax.fill_between(x, 0, solar, color=SOLAR_FILL, alpha=0.30, linewidth=0)
    ax.plot(x, solar, color=SOLAR_LINE, linewidth=2.0)
    _style(ax, "Solar generation  (MW-DC)")
    ax.set_ylim(0, max(solar.max() * 1.12, 1.0))

    # -- rows 2..n: one controller each ------------------------------------
    power_max = max(
        float(run.hourly["unconstrained_demand_mw"].to_numpy(dtype=float)[sl].max())  # type: ignore[attr-defined]
        for run in runs.values()
    )
    for i, name in enumerate(names):
        run = runs[name]
        ax = axes[1 + i]
        color = _color(name, i)
        demand = run.hourly["unconstrained_demand_mw"].to_numpy(dtype=float)[sl]  # type: ignore[attr-defined]
        target = run.hourly["controller_target_it_mw"].to_numpy(dtype=float)[sl]  # type: ignore[attr-defined]
        delivered = run.hourly["delivered_it_mw"].to_numpy(dtype=float)[sl]  # type: ignore[attr-defined]

        # voluntary gap: demand down to target
        ax.fill_between(x, target, demand, color=INK_MUTED, alpha=0.22, linewidth=0, step="mid")
        # involuntary gap: target down to delivered
        ax.fill_between(x, delivered, target, color="#d03b3b", alpha=0.35, linewidth=0, step="mid")
        # delivered body
        ax.fill_between(x, 0, delivered, color=color, alpha=0.20, linewidth=0, step="mid")

        ax.step(x, demand, where="mid", color=INK_SECONDARY, linewidth=1.4, linestyle=(0, (4, 3)))
        ax.step(x, target, where="mid", color=color, linewidth=2.0)
        ax.step(x, delivered, where="mid", color=color, linewidth=2.0, linestyle=(0, (1, 1.6)))

        metrics = run.metrics  # type: ignore[attr-defined]
        _style(
            ax,
            f"{name} — GPU power  (MW)"
            f"      {metrics['compute_units']:,.0f} compute-unit-hours"
            f"  ·  {metrics['involuntary_shortfall_mwh']:,.0f} MWh unserved"
            f"  ·  {metrics['hours_throttled']:,} h throttled",
        )
        ax.set_ylim(0, power_max * 1.20)

    # -- last row: the consequence ----------------------------------------
    ax = axes[-1]
    capacity = float(first.metadata["sizing"]["battery_mwh"])  # type: ignore[attr-defined]
    for i, name in enumerate(names):
        soc = runs[name].hourly["battery_soc_mwh"].to_numpy(dtype=float)[sl]  # type: ignore[attr-defined]
        # The window minimum goes in the legend text rather than as a floating
        # annotation: with four strategies the minima cluster near zero at the
        # right-hand edge and any in-plot label collides with its neighbours.
        ax.plot(x, soc, color=_color(name, i), linewidth=2.0,
                label=f"{name}  (min {soc.min():,.0f} MWh)")
    ax.axhline(capacity, color=INK_MUTED, linewidth=1.0, linestyle=(0, (2, 3)))
    ax.text(x[2], capacity, " battery capacity", va="bottom", ha="left",
            fontsize=8.5, color=INK_MUTED)
    _style(ax, "Battery state of charge  (MWh)")
    # Extra pad so the caption-row legend below sits clear of the title.
    ax.title.set_position((0.0, 1.0))
    ax.set_title("Battery state of charge  (MWh)", loc="left", fontsize=10.5,
                 color=INK, fontweight="bold", pad=26)
    ax.set_ylim(0, capacity * 1.14)
    # Legend as a caption row above the axes: with four traces converging near
    # zero there is no reliable in-plot gap to drop it into.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=min(len(names), 4),
              frameon=False, fontsize=9.0, labelcolor=INK_SECONDARY,
              handlelength=2.2, columnspacing=1.8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:00"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
    for label in axes[-1].get_xticklabels():
        label.set_color(INK_SECONDARY)
        label.set_fontsize(9)

    sizing = first.metadata["sizing"]  # type: ignore[attr-defined]
    curve = first.metadata["gpu"]["curve"]  # type: ignore[attr-defined]
    fig.suptitle(
        f"Hardest {window_hours} hours of the year — "
        f"{site.scenario.location.replace('_', ' ').title()}, "
        f"{site.scenario.total_gpus:,} GPUs",
        x=0.062, y=0.985, ha="left", fontsize=15.5, color=INK, fontweight="bold",
    )
    fig.text(
        0.062, 0.958,
        f"{sizing['solar_mw_dc']:.0f} MW-DC solar · {sizing['battery_mw']:.0f} MW / "
        f"{sizing['battery_mwh']:.0f} MWh battery · identical weather, hardware and demand for every strategy · "
        f"GPU curve: {curve['name']} [{curve['kind']}]",
        ha="left", fontsize=9.5, color=INK_SECONDARY,
    )
    # One shared legend: both controller panels use the same encoding.
    fig.legend(
        handles=[
            Line2D([], [], color=INK_SECONDARY, lw=1.5, ls=(0, (4, 3)), label="unconstrained demand"),
            Line2D([], [], color=INK_MUTED, lw=2.0, label="controller target"),
            Line2D([], [], color=INK_MUTED, lw=2.0, ls=(0, (1, 1.6)), label="delivered"),
            Patch(facecolor=INK_MUTED, alpha=0.22, label="voluntary throttle (a choice)"),
            Patch(facecolor="#d03b3b", alpha=0.35, label="involuntary shortfall (a failure)"),
        ],
        loc="upper left", bbox_to_anchor=(0.060, 0.938), ncol=5, frameon=False,
        fontsize=9.5, labelcolor=INK_SECONDARY, handlelength=2.6, columnspacing=2.0,
    )
    fig.text(
        0.062, 0.012,
        "Target and delivered are drawn in each controller's own colour; where one line hides "
        "another they are equal — which is the point.",
        ha="left", fontsize=8.5, color=INK_MUTED,
    )

    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return start, end


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=6)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)


def plot_derate_sweep(
    sweep: "Mapping[float, Mapping[str, object]]",
    *,
    site,
    path: Path,
    curve_meta: dict | None = None,
) -> None:
    """Compute retained as the plant shrinks, and the advantage over a fixed load.

    Two panels, one measure each (never a dual axis): absolute compute on the
    left, and the same data re-expressed as advantage over ``fixed_load`` on the
    right. The second panel is the one that carries the argument -- it shows the
    advantage of flexible operation *growing* as capital is removed, which is
    the project's premise in one line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = sorted(sweep)
    present = set(next(iter(sweep.values())))
    # Report in the logical order (no control -> naive -> optimal), not whatever
    # order the dict or a JSON round-trip happened to produce.
    names = [n for n in STRATEGY_DISPLAY_ORDER if n in present]
    names += [n for n in sorted(present) if n not in names]

    fig, (ax_abs, ax_rel) = plt.subplots(
        1, 2, figsize=(13.0, 6.4),
        gridspec_kw={"wspace": 0.18, "top": 0.735, "bottom": 0.185,
                     "left": 0.065, "right": 0.985},
    )
    fig.patch.set_facecolor(SURFACE)

    def compute(name: str, scale: float) -> float:
        return float(sweep[scale][name].metrics["compute_units"])  # type: ignore[attr-defined]

    for i, name in enumerate(names):
        color = _color(name, i)
        ys = [compute(name, s) for s in scales]
        ax_abs.plot(scales, ys, color=color, linewidth=2.0, marker="o",
                    markersize=5, label=name)
        rel = [100 * (compute(name, s) / compute("fixed_load", s) - 1) for s in scales]
        ax_rel.plot(scales, rel, color=color, linewidth=2.0, marker="o",
                    markersize=5, label=name)

    _style(ax_abs, "Useful compute  (compute-unit-hours per year)")
    _style(ax_rel, "Advantage over a fixed load  (%)")
    ax_rel.axhline(0.0, color=INK_MUTED, linewidth=1.0)
    for ax in (ax_abs, ax_rel):
        ax.set_xlabel("infrastructure scale  (1.00 = fixed-load-optimal sizing)",
                      fontsize=9.5, color=INK_SECONDARY, labelpad=8)
        ax.grid(True, axis="x", color=GRID, linewidth=0.9)
    fig.suptitle(
        "Flexible operation is worth more the less you build",
        x=0.065, y=0.972, ha="left", fontsize=15.5, color=INK, fontweight="bold",
    )
    subtitle = (
        f"{site.scenario.location.replace('_', ' ').title()}, "
        f"{site.scenario.total_gpus:,} GPUs · solar and battery scaled together · "
        "cyclic SOC · identical weather for every strategy"
    )
    if curve_meta:
        subtitle += f"\nGPU curve: {curve_meta['name']} [{curve_meta['kind']}]"
    fig.text(0.065, 0.905, subtitle, ha="left", va="top", fontsize=9.5,
             color=INK_SECONDARY, linespacing=1.5)
    handles, labels = ax_abs.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.063, 0.845),
               ncol=len(names), frameon=False, fontsize=9.5,
               labelcolor=INK_SECONDARY, handlelength=2.4, columnspacing=2.2)
    fig.text(
        0.065, 0.018,
        "At the fixed-load-optimal sizing 73% of solar is curtailed, so energy does not bind and no "
        "policy can win. De-rate the plant and stored energy starts to matter.\n"
        "perfect_foresight_mpc edges above the annual ceiling below scale 0.35 only by accepting "
        "7.3 MWh of brownout — see ASSUMPTIONS B11.",
        ha="left", va="bottom", fontsize=8.5, color=INK_MUTED, linespacing=1.6,
    )

    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def plot_forecast_sweep(
    data: Mapping,
    *,
    site,
    path: Path,
    scale: str = "0.40",
    curve_meta: dict | None = None,
    perfect_horizon: Mapping[str, float] | None = None,
) -> None:
    """What forecast error costs, and whether lookahead still pays once it is wrong.

    Two panels, one question each. The left panel is the headline: compute
    against forecast skill, with the perfect-foresight ceiling drawn as a line
    rather than as another series, because it is a *bound* and not a policy
    anyone could run. The right panel re-tests the horizon claim -- the earlier
    result that 24 hours of lookahead captures almost everything was measured
    with a *perfect* 24-hour forecast, and a longer horizon now buys more
    foresight and more error at once.

    Forecast error is plotted as realised nRMSE at 24-hour lead rather than as
    the model's ``sigma_24h`` parameter, so the x-axis is a quantity a
    forecaster would recognise (ASSUMPTIONS B13).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["scales"][scale]["runs"]
    sizing = data["scales"][scale]["sizing"]
    fixed = rows["fixed_load"]["compute_units"]
    fixed_shortfall = rows["fixed_load"]["involuntary_shortfall_mwh"]
    ceiling = rows["perfect_foresight_annual"]["compute_units"]
    perfect_mpc = rows["perfect_foresight_mpc"]["compute_units"]
    reference_sigma = data["reference_sigma_24h"]

    noisy = [r for k, r in rows.items() if k.startswith("forecast_mpc_")]
    primary = sorted(
        (r for r in noisy if r["seed"] == data["seed"]), key=lambda r: r["sigma_24h"]
    )
    alternates = [r for r in noisy if r["seed"] != data["seed"]]

    fig, (ax_err, ax_hz) = plt.subplots(
        1, 2, figsize=(13.0, 6.4),
        gridspec_kw={"wspace": 0.18, "top": 0.735, "bottom": 0.185,
                     "left": 0.075, "right": 0.985},
    )
    fig.patch.set_facecolor(SURFACE)

    # -- left: the cost of not knowing ---------------------------------------
    x = [r["forecast_nrmse_24h_pct_of_capacity"] for r in primary]
    y = [r["compute_units"] for r in primary]
    span = (max(x) - min(x)) or 1.0

    left, right = min(x) - 0.08 * span, max(x) + 0.30 * span
    ax_err.set_xlim(left, right)

    def reference(value: float, label: str, color: str, style: str, va: str) -> None:
        """A bound is a line, not a series -- nobody can run the ceiling."""
        ax_err.axhline(value, color=color, linewidth=1.5, linestyle=style)
        ax_err.text(right, value, label + " ", ha="right", va=va, fontsize=9,
                    color=color)

    reference(ceiling, "perfect-foresight ceiling (annual LP)",
              CONTROLLER_COLORS["perfect_foresight_annual"], "--", "bottom")
    reference(perfect_mpc, "perfect-foresight MPC, 48 h",
              CONTROLLER_COLORS["perfect_foresight_mpc"], ":", "top")
    reference(fixed, "no control (fixed load)", INK_MUTED, "-", "bottom")

    ax_err.plot(x, y, color=FALLBACK_COLORS[0], linewidth=2.0, marker="o",
                markersize=6, label="forecast_mpc (48 h)", zorder=3)
    if alternates:
        ax_err.scatter(
            [r["forecast_nrmse_24h_pct_of_capacity"] for r in alternates],
            [r["compute_units"] for r in alternates],
            facecolor=SURFACE, edgecolor=FALLBACK_COLORS[0], linewidth=1.4,
            s=44, marker="D", zorder=4, label="same error, different realisation",
        )
    _style(ax_err, "Useful compute  (compute-unit-hours per year)")
    ax_err.set_xlabel(
        "forecast error  (realised nRMSE at 24 h lead, % of plant capacity)",
        fontsize=9.5, color=INK_SECONDARY, labelpad=8,
    )
    ax_err.grid(True, axis="x", color=GRID, linewidth=0.9)
    ax_err.legend(loc="lower left", frameon=False, fontsize=9,
                  labelcolor=INK_SECONDARY, handlelength=2.4)

    # -- right: does lookahead still pay when it is wrong? -------------------
    swept = data.get("horizon_sweep") or {}
    if swept.get("runs"):
        horizons = sorted(swept["runs"].values(), key=lambda r: r["horizon_h"])
        hx = [r["horizon_h"] for r in horizons]
        hy = [r["compute_units"] for r in horizons]
        ax_hz.plot(hx, hy, color=FALLBACK_COLORS[0], linewidth=2.0, marker="o",
                   markersize=6, zorder=3,
                   label=f"forecast_mpc  (nRMSE {_nrmse_at(primary, reference_sigma):.1f}%)")
        if perfect_horizon:
            px = sorted(float(h) for h in perfect_horizon)
            ax_hz.plot(px, [perfect_horizon[h] for h in sorted(perfect_horizon)],
                       color=CONTROLLER_COLORS["perfect_foresight_mpc"],
                       linewidth=2.0, marker="o", markersize=5, linestyle=":",
                       label="perfect foresight")
        ax_hz.axhline(ceiling, color=CONTROLLER_COLORS["perfect_foresight_annual"],
                      linewidth=1.6, linestyle="--")
        ax_hz.axhline(fixed, color=INK_MUTED, linewidth=1.4)
        ax_hz.legend(loc="lower right", frameon=False, fontsize=9,
                     labelcolor=INK_SECONDARY, handlelength=2.4)
    _style(ax_hz, "Compute vs lookahead, at a fixed forecast error")
    ax_hz.set_xlabel("MPC lookahead  (hours)", fontsize=9.5,
                     color=INK_SECONDARY, labelpad=8)
    ax_hz.grid(True, axis="x", color=GRID, linewidth=0.9)

    fig.suptitle(
        "What a wrong forecast costs",
        x=0.075, y=0.972, ha="left", fontsize=15.5, color=INK, fontweight="bold",
    )
    subtitle = (
        f"{site.scenario.location.replace('_', ' ').title()}, "
        f"{site.scenario.total_gpus:,} GPUs · scale {scale} "
        f"({sizing['solar_mw_dc']:.0f} MW-DC / {sizing['battery_mw']:.0f} MW / "
        f"{sizing['battery_mwh']:.0f} MWh) · same solver, same plant, "
        "belief instead of truth"
    )
    if curve_meta:
        subtitle += f"\nGPU curve: {curve_meta['name']} [{curve_meta['kind']}]"
    fig.text(0.075, 0.895, subtitle, ha="left", va="top", fontsize=9.5,
             color=INK_SECONDARY, linespacing=1.5)
    # Compute is never shown without reliability beside it: a controller can
    # always buy compute by browning out (ASSUMPTIONS B11), so the figure has to
    # state what these points cost in unserved energy rather than imply zero.
    worst = max((r["involuntary_shortfall_mwh"] for r in noisy), default=0.0)
    reliability = (
        "Every point delivered what it asked for: zero involuntary shortfall, so none of this "
        "compute was bought by browning out."
        if worst <= 1e-6 else
        f"Worst involuntary shortfall across these runs: {worst:.1f} MWh/yr, against "
        f"{fixed_shortfall:,.0f} MWh/yr for the fixed load — so the compute is not bought by "
        "browning out."
    )
    fig.text(
        0.075, 0.018,
        "Error model is synthetic and its parameters are chosen, not fitted — read the shape of the "
        f"response, not the value at one error level (ASSUMPTIONS B13).\n{reliability}",
        ha="left", va="bottom", fontsize=8.5, color=INK_MUTED, linespacing=1.6,
    )

    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)


def _nrmse_at(rows, sigma: float) -> float:
    for row in rows:
        if abs(row["sigma_24h"] - sigma) < 1e-9:
            return row["forecast_nrmse_24h_pct_of_capacity"]
    return float("nan")
