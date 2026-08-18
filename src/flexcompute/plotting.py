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
# surface with pairs="all" (not merely adjacent, because a six-entry legend shows
# every slot at once): worst all-pairs CVD ΔE 9.2 (deutan), tritan 9.0,
# normal-vision ΔE 16.3, all six inside the lightness band. Aqua sits below 3:1
# contrast, so the relief rule applies -- every aqua line carries a visible
# direct label. Do not substitute without re-running scripts/validate_palette.js.
#
# The two slots added for the six-rung ladder cost nothing: the worst pair is
# still the pre-existing aqua/orange one, so extending the palette did not
# degrade it.
#
# Hue assignment carries meaning as well as identity. ``casey_governor`` is a
# deep sienna, deliberately in the same warm family as ``simple_throttle``'s
# orange, because the two are the forecast-free heuristics and their kinship is
# real. ``forecast_mpc`` is magenta, as far as possible from
# ``perfect_foresight_mpc``'s aqua, because *that* pair is the comparison the
# whole study turns on and it must never be misread.
CONTROLLER_COLORS = {
    "fixed_load": "#2a78d6",                # blue
    "simple_throttle": "#eb6834",           # orange
    "casey_governor": "#9e441d",            # sienna
    "forecast_mpc": "#b24f9e",              # magenta
    "perfect_foresight_mpc": "#1baf7a",     # aqua
    "perfect_foresight_annual": "#4a3aa7",  # violet
}
FALLBACK_COLORS = ["#e87ba4", "#008300", "#e34948", "#eda100"]

#: Human-readable names, so a legend never shows a Python identifier.
CONTROLLER_LABELS = {
    "fixed_load": "fixed load",
    "simple_throttle": "simple throttle",
    "casey_governor": "Casey governor",
    "forecast_mpc": "forecast MPC",
    "perfect_foresight_mpc": "perfect-foresight MPC",
    "perfect_foresight_annual": "perfect foresight (annual)",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8c8b86"
GRID = "#e6e5e1"
SOLAR_FILL = "#eda100"
SOLAR_LINE = "#a87200"

REFERENCE_YEAR = 2023

#: Logical reporting order: no control -> forecast-free -> forecast-aware -> ceiling.
STRATEGY_DISPLAY_ORDER = (
    "fixed_load",
    "simple_throttle",
    "casey_governor",
    "forecast_mpc",
    "perfect_foresight_mpc",
    "perfect_foresight_annual",
)


def _color(name: str, index: int) -> str:
    return CONTROLLER_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def _label(name: str) -> str:
    return CONTROLLER_LABELS.get(name, name.replace("_", " "))


def _ordered(names) -> list[str]:
    """Sort strategy names into the reporting order, unknowns last."""
    order = {name: i for i, name in enumerate(STRATEGY_DISPLAY_ORDER)}
    return sorted(names, key=lambda n: (order.get(n, len(order)), n))


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
    start_hour: int | None = None,
    title: str | None = None,
    subtitle: str | None = None,
) -> tuple[int, int]:
    """Render the difficult-window figure. Returns the (start, end) hours.

    ``start_hour`` pins the window instead of letting it be chosen by controller
    stress. That matters for the multi-year headline figure, where the window
    must be the hardest *solar drought in the record* -- a property of the
    weather, identical for every strategy -- rather than the window where some
    particular controller struggled most. Choosing the window by stress would
    pick a different span for a different strategy set, which is not a fair
    place to compare them.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if start_hour is None:
        start, end = find_difficult_window(runs, window_hours)
    else:
        start, end = int(start_hour), int(start_hour) + window_hours
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

        # Scored over the window on screen, not over the year. An annual total
        # printed above a five-day panel invites the reader to attribute the
        # whole year's shortfall to what they can see.
        metrics = run.metrics  # type: ignore[attr-defined]
        window_compute = float(run.hourly["compute_units"].to_numpy(dtype=float)[sl].sum())  # type: ignore[attr-defined]
        window_short = float(
            run.hourly["involuntary_shortfall_mw"].to_numpy(dtype=float)[sl].sum()  # type: ignore[attr-defined]
        )
        _style(
            ax,
            f"{_label(name)} — GPU power  (MW)"
            f"     this window: {window_compute:,.1f} cu-h, "
            f"{window_short:,.0f} MWh unserved"
            f"     ·     full year: {metrics['compute_units']:,.0f} cu-h, "
            f"{metrics['involuntary_shortfall_mwh']:,.0f} MWh",
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
                label=f"{_label(name)}  (min {soc.min():,.0f} MWh)")
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
        title or (
            f"Hardest {window_hours} hours of the year — "
            f"{site.scenario.location.replace('_', ' ').title()}, "
            f"{site.scenario.total_gpus:,} GPUs"
        ),
        x=0.062, y=0.985, ha="left", fontsize=15.5, color=INK, fontweight="bold",
    )
    fig.text(
        0.062, 0.958,
        subtitle or (
            f"{sizing['solar_mw_dc']:.0f} MW-DC solar · {sizing['battery_mw']:.0f} MW / "
            f"{sizing['battery_mwh']:.0f} MWh battery · identical weather, hardware and "
            f"demand for every strategy · GPU curve: {curve['name']} [{curve['kind']}]"
        ),
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


# ---------------------------------------------------------------------------
# The multi-year headline figures
# ---------------------------------------------------------------------------
#
# Shared conventions, applied to all five:
#
# * one measure per axis, never a second y-scale;
# * a controller keeps its hue everywhere, so a reader learns the key once and
#   it survives a change of figure or a change of which series are present;
# * every series that is a line is also directly labelled, because the aqua slot
#   sits below 3:1 contrast against the surface and because identity must never
#   rest on colour alone;
# * spread is drawn wherever the underlying claim is about a distribution --
#   a median line with no band would assert a confidence the 15 years do not
#   support.


def _spread_labels(values: "list[float]", *, min_gap: float) -> "list[float]":
    """Nudge end-of-line label positions apart, preserving their order.

    Direct labels are mandatory here -- one palette slot sits below 3:1 contrast
    and identity must never rest on colour alone -- but several strategies
    converge to within a fraction of a percent at the right-hand edge, so the
    labels collide exactly where the reader most needs them. This spreads them
    to a minimum spacing while keeping the vertical ordering, so a label still
    sits nearest its own line.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for rank, i in enumerate(order):
        if rank == 0:
            continue
        previous = placed[order[rank - 1]]
        if placed[i] - previous < min_gap:
            placed[i] = previous + min_gap
    return placed


def _finish(fig, path: Path, *, caption: str | None = None) -> Path:
    if caption:
        fig.text(0.062, 0.012, caption, ha="left", va="bottom",
                 fontsize=8.2, color=INK_MUTED, wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _new_figure(figsize, **gridspec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(figsize=figsize, **gridspec)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def plot_controller_value_vs_scarcity(
    aggregates: Mapping[str, Mapping[str, dict]],
    per_year: Mapping[str, Mapping[str, Mapping[str, dict]]],
    *,
    path: Path,
    n_years: int,
) -> Path:
    """Figure 1 — how much control is worth as energy gets scarce.

    Median advantage over fixed load against infrastructure scale, with a
    P10-P90 band across the weather years. The band is the point as much as the
    line: at a generous sizing every policy is within noise of every other, and
    a median-only chart would make that look like a finding rather than an
    absence of one.

    x runs from generous to scarce (left to right) so the reader travels in the
    direction of the argument.
    """
    scales = sorted((float(s) for s in aggregates), reverse=True)
    strategies = _ordered(
        {s for scale in aggregates.values() for s in scale if s != "fixed_load"}
    )

    fig, ax = _new_figure((12.2, 6.4),
                          gridspec_kw={"top": 0.86, "bottom": 0.155,
                                       "left": 0.070, "right": 0.735})
    x = np.arange(len(scales))

    ends: list[tuple[str, float, str]] = []
    for i, strategy in enumerate(strategies):
        med, lo, hi = [], [], []
        for scale in scales:
            stats = aggregates[str(scale)].get(strategy, {}).get(
                "advantage_pct_vs_fixed", {})
            med.append(stats.get("median", np.nan))
            lo.append(stats.get("p10", np.nan))
            hi.append(stats.get("p90", np.nan))
        color = _color(strategy, i)
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.plot(x, med, color=color, linewidth=2.0, solid_capstyle="round",
                marker="o", markersize=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.6, zorder=3 + i)
        ends.append((strategy, float(med[-1]), color))

    ax.axhline(0.0, color=INK_SECONDARY, linewidth=1.2, zorder=2)

    # Direct labels, spread apart and leadered back to their own line so that
    # the three near-identical foresight curves stay individually readable.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    placed = _spread_labels([e[1] for e in ends], min_gap=0.062 * span)
    for (strategy, y, color), y_label in zip(ends, placed):
        ax.annotate(
            _label(strategy), xy=(x[-1], y), xytext=(x[-1] + 0.16, y_label),
            va="center", ha="left", fontsize=9.5, color=color, fontweight="bold",
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=color, linewidth=0.9,
                            shrinkA=0, shrinkB=2, alpha=0.55),
        )
    ax.annotate("fixed load (baseline)", xy=(x[-1], 0.0),
                xytext=(x[-1] + 0.16, 0.022 * span), va="center", ha="left",
                fontsize=9.5, color=INK_SECONDARY, fontweight="bold",
                annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.2f}" for s in scales])
    ax.set_xlabel("infrastructure scale  (1.00 = the reference plant; "
                  "solar MW, battery MW and MWh all scaled together)",
                  fontsize=9.5, color=INK_SECONDARY, labelpad=9)
    ax.set_ylabel("useful compute vs fixed load  (%)", fontsize=9.5,
                  color=INK_SECONDARY)
    _style(ax, "Control is worth almost nothing until energy is scarce")
    fig.text(0.075, 0.925,
             f"median across {n_years} actual Dallas weather years; "
             f"band spans P10-P90",
             fontsize=9.5, color=INK_SECONDARY)

    return _finish(fig, path, caption=(
        "Infrastructure scale is an Experiment A device: the same 10,000 GPUs on a "
        "smaller plant. It is not a claim that any scale is economically preferable — "
        "Experiment B decides sizing. Compute alone can be bought with brownouts; see "
        "the shortfall columns in the results JSON."))


def plot_year_distribution(
    per_year: Mapping[str, Mapping[str, Mapping[str, dict]]],
    *,
    path: Path,
    scale: float,
    concentration: Mapping[str, dict] | None = None,
) -> Path:
    """Figure 5 — is the advantage every year, or two bad years?

    One dot per weather year per controller, with the median marked. A strip
    plot rather than a box plot: with fifteen points the raw sample is small
    enough to show outright, and hiding it behind quartiles would obscure
    exactly the question being asked.

    Years are the unit of observation, so each dot is labelled by its year where
    the drought years are — those are the ones a reader wants to identify.
    """
    key = str(scale)
    years = sorted(per_year)
    strategies = _ordered(
        {s for y in years for s in per_year[y].get(key, {}) if s != "fixed_load"}
    )

    fig, ax = _new_figure((11.0, 0.95 * len(strategies) + 3.6),
                          gridspec_kw={"top": 0.845, "bottom": 0.145,
                                       "left": 0.235, "right": 0.965})

    rng = np.random.default_rng(7)
    for row, strategy in enumerate(strategies):
        color = _color(strategy, row)
        values, labels = [], []
        for year in years:
            entry = per_year[year].get(key, {}).get(strategy)
            if entry is None:
                continue
            values.append(entry["advantage_pct_vs_fixed"])
            labels.append(year)
        if not values:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(values, np.full(len(values), row) + jitter, s=46,
                   color=color, alpha=0.72, edgecolor=SURFACE, linewidth=1.1,
                   zorder=3)
        median = float(np.median(values))
        ax.plot([median, median], [row - 0.30, row + 0.30], color=color,
                linewidth=2.6, solid_capstyle="round", zorder=4)
        # Median value sits clear of the strip, on the far side of the widest
        # point, so it never lands on top of a dot or on its own bar.
        ax.annotate(f"median {median:+.1f}%", (max(values), row),
                    xytext=(14, 0), textcoords="offset points", va="center",
                    ha="left", fontsize=9.0, color=color, fontweight="bold")
        # Name only the extremes, outboard of their own dot, so the reader can
        # identify which weather years sit at the edges of the distribution.
        lo_i, hi_i = int(np.argmin(values)), int(np.argmax(values))
        ax.annotate(str(labels[lo_i]), (values[lo_i], row + jitter[lo_i]),
                    xytext=(-8, 0), textcoords="offset points", va="center",
                    ha="right", fontsize=7.8, color=INK_MUTED)
        ax.annotate(str(labels[hi_i]), (values[hi_i], row + jitter[hi_i]),
                    xytext=(0, 11), textcoords="offset points", va="bottom",
                    ha="center", fontsize=7.8, color=INK_MUTED)

    ax.axvline(0.0, color=INK_SECONDARY, linewidth=1.2, zorder=2)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels([_label(s) for s in strategies], fontsize=10)
    ax.set_ylim(-0.7, len(strategies) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("useful compute vs fixed load, one point per weather year  (%)",
                  fontsize=9.5, color=INK_SECONDARY, labelpad=9)
    _style(ax, f"Every weather year, at infrastructure scale {scale:.2f}")
    ax.grid(True, axis="x", color=GRID, linewidth=0.9)
    ax.grid(False, axis="y")
    # Headroom both sides: the median label sits outboard right, the min-year
    # label outboard left where it would otherwise land on the y tick labels.
    x_lo, x_hi = ax.get_xlim()
    span_x = x_hi - x_lo
    ax.set_xlim(x_lo - 0.05 * span_x, x_hi + 0.19 * span_x)

    # One short summary rather than a per-strategy list: the per-strategy shares
    # were identical to the percentage point, and a line long enough to hold all
    # of them runs off the figure.
    note = " "
    if concentration:
        shares = [c["top_3_year_share"] for c in concentration.values()
                  if c.get("interpretable")]
        if shares:
            even = next(iter(concentration.values())).get("even_split_top_3_share")
            band = (f"{min(shares):.0%}" if round(min(shares), 2) == round(max(shares), 2)
                    else f"{min(shares):.0%}–{max(shares):.0%}")
            note = (f"The three best years hold {band} of each controller's total gain "
                    f"({even:.0%} would be a perfectly even spread).")
    fig.text(0.235, 0.912, note, fontsize=9.2, color=INK_SECONDARY)

    return _finish(fig, path, caption=(
        "Tightly clustered dots mean the advantage is a property of the climate; a wide "
        "fan means it is carried by a few bad years and the mean overstates a typical "
        "one. Compute alone can be bought with brownouts — see the shortfall columns "
        "in the results JSON."))


def plot_forecast_error_sensitivity(
    rows: "list[dict]",
    *,
    path: Path,
    fixed_load_compute: float,
    ceiling_compute: float,
    scale: float,
    n_years: int,
    per_year: "list[list[float]] | None" = None,
) -> Path:
    """Figure 3 — what a wrong forecast costs, in realised-error units.

    x is **realised day-ahead nRMSE**, an error magnitude normalised by plant
    capacity. It is not a failure rate and the axis label says so, because that
    misreading has happened before.

    Two horizontal references bracket the achievable range: doing nothing, and
    knowing the future exactly. Without them a reader cannot tell whether a
    given drop matters, since the whole vertical range at stake is a couple of
    percent.
    """
    rows = sorted(rows, key=lambda r: r["nrmse"])
    x = [r["nrmse"] for r in rows]
    med = [r["median"] for r in rows]

    fig, ax = _new_figure((11.4, 6.3),
                          gridspec_kw={"top": 0.855, "bottom": 0.175,
                                       "left": 0.090, "right": 0.735})

    color = _color("forecast_mpc", 3)

    # One faint line per weather year rather than a P10-P90 band. The years sit
    # at different absolute levels -- a cloudy year simply produces less -- so a
    # band drawn across them measures the *climate's* spread, not the
    # uncertainty in what forecast error costs, and would swamp the effect being
    # shown. The per-year lines are paired: each one is the same year at four
    # error levels, so their common downward slope is the actual finding.
    if per_year:
        for series in per_year:
            ax.plot(x, series, color=color, linewidth=0.9, alpha=0.30, zorder=2)

    ax.plot(x, med, color=color, linewidth=2.6, marker="o", markersize=6.5,
            markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=5)

    labels = [(med[-1], f"{_label('forecast_mpc')} (median)", color)]
    for value, name, hue in (
        (ceiling_compute, "perfect foresight, annual (median)",
         _color("perfect_foresight_annual", 5)),
        (fixed_load_compute, "fixed load (median)", _color("fixed_load", 0)),
    ):
        ax.axhline(value, color=hue, linewidth=1.6, linestyle=(0, (5, 3)), zorder=3)
        labels.append((value, name, hue))

    # Place every right-hand label outside the plot area and spread them, so a
    # label never sits on top of the line it names.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    placed = _spread_labels([v for v, _, _ in labels], min_gap=0.058 * span)
    for (value, name, hue), y in zip(labels, placed):
        ax.annotate(name, xy=(x[-1], value),
                    xytext=(x[-1] + 0.055 * (x[-1] - x[0]) + 0.4, y),
                    va="center", ha="left", fontsize=9.5, color=hue,
                    fontweight="bold", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=hue, linewidth=0.9,
                                    shrinkA=0, shrinkB=2, alpha=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}%" for v in x])
    ax.set_xlabel("realised day-ahead forecast error  (nRMSE, % of plant capacity)\n"
                  "an error magnitude — NOT a fraction of hours forecast wrongly",
                  fontsize=9.5, color=INK_SECONDARY, labelpad=9)
    ax.set_ylabel("useful compute  (compute-unit-hours)", fontsize=9.5,
                  color=INK_SECONDARY)
    _style(ax, "What not knowing the weather costs")
    fig.text(0.095, 0.925,
             f"infrastructure scale {scale:.2f}, where stored energy binds; "
             f"one faint line per weather year, bold line is the median of {n_years}",
             fontsize=9.5, color=INK_SECONDARY)

    return _finish(fig, path, caption=(
        "The synthetic forecast-error model is not a validated forecasting system "
        "(ASSUMPTIONS B13). Read the shape of the response across the sweep, not the "
        "value at any single point."))


def plot_economic_comparison(
    designs: "list[dict]",
    *,
    path: Path,
    reference_capex: float,
    variant_label: str,
) -> Path:
    """Figure 4 — capital at equal compute and equal reliability.

    Two stacked panels sharing one categorical x: capital on top, the design
    that buys it below. Two panels rather than one chart with a second y-axis,
    which is the rule; and the sizing panel is what stops the cost bar being
    read as a free lunch.
    """
    designs = list(designs)
    names = [d["strategy"] for d in designs]
    x = np.arange(len(designs))

    fig, (ax_cost, ax_size) = _new_figure(
        (10.8, 8.4), nrows=2, sharex=True,
        gridspec_kw={"hspace": 0.30, "top": 0.875, "bottom": 0.115,
                     "left": 0.085, "right": 0.965, "height_ratios": [1.25, 1.0]},
    )

    # -- capital -----------------------------------------------------------
    colors = [_color(n, i) for i, n in enumerate(names)]
    bars = ax_cost.bar(x, [d["capex_musd"] for d in designs], width=0.58,
                       color=colors, linewidth=0)
    for bar in bars:
        bar.set_edgecolor(SURFACE)
        bar.set_linewidth(2.0)
    baseline = designs[0]["capex_musd"]
    for i, d in enumerate(designs):
        delta = 100.0 * (d["capex_musd"] / baseline - 1.0)
        label = f"{d['capex_musd']:,.0f} M$"
        if i:
            label += f"\n{delta:+.1f}% vs fixed load"
        ax_cost.annotate(label, (i, d["capex_musd"]), xytext=(0, 7),
                         textcoords="offset points", ha="center",
                         fontsize=9.2, color=INK_SECONDARY, linespacing=1.35)
    ax_cost.axhline(reference_capex, color=INK_MUTED, linewidth=1.4,
                    linestyle=(0, (5, 3)), zorder=1)
    ax_cost.annotate(f"reference plant sized for 99% uptime  ({reference_capex:,.0f} M$)",
                     (len(designs) - 0.45, reference_capex), xytext=(0, 5),
                     textcoords="offset points", ha="right", fontsize=8.8,
                     color=INK_MUTED)
    ax_cost.set_ylabel("year-0 solar + BESS CAPEX  (M$, 2022)", fontsize=9.5,
                       color=INK_SECONDARY)
    ax_cost.set_ylim(0, max(max(d["capex_musd"] for d in designs),
                            reference_capex) * 1.26)
    _style(ax_cost, f"Capital for the same compute — {variant_label}")

    # -- the design that buys it -------------------------------------------
    width = 0.26
    series = (
        ("solar MW-DC", [d["solar_mw"] for d in designs], "#a87200"),
        ("battery MW", [d["battery_mw"] for d in designs], INK_SECONDARY),
        ("battery MWh / 10", [d["battery_mwh"] / 10.0 for d in designs], "#6f6d68"),
    )
    for k, (name, values, hue) in enumerate(series):
        offset = (k - 1) * width
        b = ax_size.bar(x + offset, values, width=width * 0.9, color=hue,
                        linewidth=2.0, edgecolor=SURFACE, label=name)
        for xi, v in zip(x + offset, values):
            ax_size.annotate(f"{v:,.0f}", (xi, v), xytext=(0, 4),
                             textcoords="offset points", ha="center",
                             fontsize=8.0, color=INK_SECONDARY)
    for i, d in enumerate(designs):
        flag = " !" if d.get("duration_extrapolated") else ""
        ax_size.annotate(f"{d['duration_h']:.1f} h duration{flag}", (i, 0),
                         xytext=(0, -26), textcoords="offset points",
                         ha="center", fontsize=8.6,
                         color="#b3541e" if flag else INK_MUTED)
    ax_size.set_ylabel("MW  ·  MWh/10", fontsize=9.5, color=INK_SECONDARY)
    _style(ax_size, "The design that buys it")
    ax_size.legend(frameon=False, fontsize=9, ncol=3, loc="upper right",
                   labelcolor=INK_SECONDARY)
    ax_size.set_xticks(x)
    ax_size.set_xticklabels([_label(n) for n in names], fontsize=10)
    ax_size.tick_params(axis="x", pad=18)

    caption = ("Year-0 capital only: no O&M, no battery replacement, no degradation, "
               "no discounting. Not an LCOE.")
    if any(d.get("duration_extrapolated") for d in designs):
        caption += ("  '!' marks a duration outside the 2-10 h range the battery "
                    "cost split's source data spans — those figures extrapolate it.")
    return _finish(fig, path, caption=caption)
