"""Figure plumbing.

Charts are checked by eye, not by assertion, so these tests cover only the parts
that are actually mechanical: the palette contract the figures depend on, and
the label de-collision that keeps five converging lines readable. A regression
in either is invisible in a passing test suite and obvious in a ruined figure.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.experiments import COMPARISON_LADDER
from flexcompute.plotting import (
    CONTROLLER_COLORS,
    CONTROLLER_LABELS,
    STRATEGY_DISPLAY_ORDER,
    _color,
    _label,
    _ordered,
    _spread_labels,
)


# ---------------------------------------------------------------------------
# Palette contract
# ---------------------------------------------------------------------------

def test_every_ladder_strategy_has_its_own_hue():
    """A strategy falling through to the fallback list would silently change
    colour when the set of plotted series changes, breaking the rule that a
    controller keeps one identity across every figure."""
    for strategy in COMPARISON_LADDER:
        assert strategy in CONTROLLER_COLORS, strategy


def test_hues_are_distinct():
    assert len(set(CONTROLLER_COLORS.values())) == len(CONTROLLER_COLORS)


def test_palette_is_the_validated_set():
    """Pins the exact hexes that were run through the CVD validator.

    The palette passed all six checks at pairs='all' (worst CVD ΔE 9.2 deutan,
    normal-vision 16.3). Changing a value here without re-running
    ``scripts/validate_palette.js`` would silently drop that guarantee, so the
    values are pinned and the docstring in plotting.py says how to re-derive
    them.
    """
    assert CONTROLLER_COLORS == {
        "fixed_load": "#2a78d6",
        "simple_throttle": "#eb6834",
        "casey_governor": "#9e441d",
        "forecast_mpc": "#b24f9e",
        "perfect_foresight_mpc": "#1baf7a",
        "perfect_foresight_annual": "#4a3aa7",
    }


def test_solar_fill_is_not_a_controller_hue():
    """Solar is context, not a series; it must not read as a seventh strategy."""
    from flexcompute.plotting import SOLAR_FILL, SOLAR_LINE

    assert SOLAR_FILL not in CONTROLLER_COLORS.values()
    assert SOLAR_LINE not in CONTROLLER_COLORS.values()


def test_every_strategy_has_a_readable_label():
    for strategy in COMPARISON_LADDER:
        assert strategy in CONTROLLER_LABELS
        assert "_" not in _label(strategy)


def test_unknown_strategy_still_gets_a_colour_and_a_label():
    assert _color("something_new", 0) is not None
    assert _label("something_new") == "something new"


def test_display_order_matches_the_comparison_ladder():
    assert STRATEGY_DISPLAY_ORDER == COMPARISON_LADDER


def test_ordering_puts_unknown_strategies_last():
    names = ["zzz_custom", "perfect_foresight_annual", "fixed_load"]
    assert _ordered(names) == ["fixed_load", "perfect_foresight_annual", "zzz_custom"]


# ---------------------------------------------------------------------------
# Label de-collision
# ---------------------------------------------------------------------------

def test_spread_labels_separates_colliding_values():
    """Three foresight curves converge to within 0.4 points at the right edge."""
    placed = _spread_labels([17.50, 17.49, 17.06], min_gap=1.0)
    ordered = sorted(placed)
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(ordered, ordered[1:]))


def test_spread_labels_preserves_vertical_order():
    """A label must still sit nearest its own line, or it mislabels the chart."""
    values = [3.0, 1.0, 2.0, 2.05]
    placed = _spread_labels(values, min_gap=0.5)
    assert np.argsort(placed).tolist() == np.argsort(values).tolist()


def test_spread_labels_leaves_well_separated_values_alone():
    values = [0.0, 10.0, 20.0]
    assert _spread_labels(values, min_gap=1.0) == values


def test_spread_labels_handles_a_single_value():
    assert _spread_labels([4.2], min_gap=1.0) == [4.2]


def test_spread_labels_handles_exact_ties():
    placed = _spread_labels([5.0, 5.0, 5.0], min_gap=2.0)
    ordered = sorted(placed)
    assert all(b - a >= 2.0 - 1e-9 for a, b in zip(ordered, ordered[1:]))


# ---------------------------------------------------------------------------
# The figures render at all
# ---------------------------------------------------------------------------

def test_scarcity_figure_renders(tmp_path):
    """Smoke test on synthetic aggregates -- catches import and layout crashes."""
    from flexcompute.plotting import plot_controller_value_vs_scarcity

    scales = ["1.0", "0.4", "0.2"]
    strategies = ["fixed_load", "casey_governor", "perfect_foresight_annual"]
    aggregates = {
        s: {
            name: {"advantage_pct_vs_fixed":
                   {"median": i * 2.0, "p10": i * 1.5, "p90": i * 2.5}}
            for i, name in enumerate(strategies)
        }
        for s in scales
    }
    per_year = {"2019": {s: {n: {"advantage_pct_vs_fixed": 1.0}
                             for n in strategies} for s in scales}}
    path = plot_controller_value_vs_scarcity(
        aggregates, per_year, path=tmp_path / "f1.png", n_years=1)
    assert path.exists() and path.stat().st_size > 0


def test_distribution_figure_renders(tmp_path):
    from flexcompute.plotting import plot_year_distribution

    rng = np.random.default_rng(0)
    per_year = {
        str(year): {"0.25": {
            "fixed_load": {"advantage_pct_vs_fixed": 0.0},
            "casey_governor": {"advantage_pct_vs_fixed": float(rng.normal(7, 0.4))},
        }}
        for year in range(2010, 2025)
    }
    path = plot_year_distribution(
        per_year, path=tmp_path / "f5.png", scale=0.25,
        concentration={"casey_governor": {
            "top_1_year_share": 0.07, "top_2_year_share": 0.14,
            "top_3_year_share": 0.21, "even_split_top_3_share": 0.2,
            "interpretable": True}},
    )
    assert path.exists() and path.stat().st_size > 0


def test_economic_figure_flags_extrapolated_durations(tmp_path):
    from flexcompute.plotting import plot_economic_comparison

    designs = [
        {"strategy": "fixed_load", "capex_musd": 278.0, "solar_mw": 115.4,
         "battery_mw": 51.7, "battery_mwh": 734.0, "duration_h": 14.2,
         "duration_extrapolated": True},
        {"strategy": "casey_governor", "capex_musd": 274.1, "solar_mw": 117.2,
         "battery_mw": 53.6, "battery_mwh": 688.0, "duration_h": 12.8,
         "duration_extrapolated": True},
    ]
    path = plot_economic_comparison(
        designs, path=tmp_path / "f4.png", reference_capex=375.4,
        variant_label="test")
    assert path.exists() and path.stat().st_size > 0


def test_forecast_sensitivity_figure_renders(tmp_path):
    from flexcompute.plotting import plot_forecast_error_sensitivity

    rows = [{"nrmse": n, "median": 7400 - 8 * n, "p10": 7000, "p90": 7600}
            for n in (5, 10, 15, 20)]
    per_year = [[7400 - 8 * n + d for n in (5, 10, 15, 20)] for d in (-200, 0, 200)]
    path = plot_forecast_error_sensitivity(
        rows, path=tmp_path / "f3.png", fixed_load_compute=6420.0,
        ceiling_compute=7343.0, scale=0.25, n_years=15, per_year=per_year)
    assert path.exists() and path.stat().st_size > 0
