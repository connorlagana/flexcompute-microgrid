"""Capital cost model tests (ASSUMPTIONS Q2).

The whole point of this model is that it prices battery *energy* separately from
battery *power*, which upstream does not. The risk in doing that is drift: a
decomposition that quietly changes the absolute cost level would invalidate
every comparison against the Milestone-1 baseline. So the central test is the
reconciliation — at four hours the decomposed model must return upstream's own
number, exactly.
"""

from __future__ import annotations

import pytest

from flexcompute.costs import (
    NREL_ENERGY_COST_PER_KWH,
    NREL_POWER_COST_PER_KW,
    NREL_REFERENCE_DURATION_H,
    CapitalCostModel,
    energy_share_at_reference_duration,
    from_upstream_config,
    upstream_equivalent_capex_usd,
)


@pytest.fixture(scope="module")
def model(cfg):
    return from_upstream_config(cfg, "ac_coupled")


# ---------------------------------------------------------------------------
# The reconciliation
# ---------------------------------------------------------------------------

def test_decomposition_reproduces_upstream_at_four_hours(model):
    """cost_per_kwh x 4 + cost_per_kw == upstream's quoted per-kW figure."""
    reconstructed = (
        model.storage_cost_per_kwh * NREL_REFERENCE_DURATION_H + model.storage_cost_per_kw
    )
    assert reconstructed == pytest.approx(model.storage_cost_per_kw_at_ref, rel=1e-12)


@pytest.mark.parametrize("solar_mw,battery_mw", [(199.6, 122.6), (50.0, 25.0), (400.0, 200.0)])
def test_total_capex_matches_upstream_at_four_hours(cfg, model, solar_mw, battery_mw):
    """Including land, which scales with MWh and so is sensitive to duration."""
    ours = model.total_capex_usd(solar_mw, battery_mw, battery_mw * 4.0)
    theirs = upstream_equivalent_capex_usd(cfg, solar_mw, battery_mw, "ac_coupled")
    assert ours == pytest.approx(theirs, rel=1e-12)


@pytest.mark.parametrize("architecture", ["ac_coupled", "dc_coupled"])
def test_reconciliation_holds_for_both_architectures(cfg, architecture):
    model = from_upstream_config(cfg, architecture)
    ours = model.total_capex_usd(150.0, 80.0, 320.0)
    theirs = upstream_equivalent_capex_usd(cfg, 150.0, 80.0, architecture)
    assert ours == pytest.approx(theirs, rel=1e-12)


# ---------------------------------------------------------------------------
# The source ratio
# ---------------------------------------------------------------------------

def test_published_duration_table_is_reproduced_by_the_affine_fit():
    """The slope/intercept we use must actually fit the published table.

    NLR/NREL total installed cost, 2/4/6/8/10 h: 403 / 574 / 744 / 915 / 1086
    $/kW. If our two constants do not reproduce those to within a dollar, the
    ratio we are borrowing is not the one that source published.
    """
    published = {2: 403, 4: 574, 6: 744, 8: 915, 10: 1086}
    for duration, expected in published.items():
        fitted = NREL_ENERGY_COST_PER_KWH * duration + NREL_POWER_COST_PER_KW
        assert abs(fitted - expected) <= 1.0, f"{duration} h: {fitted} vs {expected}"


def test_energy_share_is_a_sane_fraction():
    share = energy_share_at_reference_duration()
    assert 0.4 < share < 0.8      # storage-heavy but not the whole system


# ---------------------------------------------------------------------------
# Duration is no longer free
# ---------------------------------------------------------------------------

def test_longer_duration_costs_more(model):
    """The defect this model exists to fix: under upstream's $/kW-only pricing
    an optimiser could buy unlimited MWh at zero cost."""
    base = model.total_capex_usd(100.0, 50.0, 200.0)
    for duration in (5.0, 8.0, 12.0):
        assert model.total_capex_usd(100.0, 50.0, 50.0 * duration) > base


def test_power_and_energy_are_independently_priced(model):
    """Doubling MW alone and doubling MWh alone must both cost, differently."""
    base = model.storage_capex_usd(50.0, 200.0)
    more_power = model.storage_capex_usd(100.0, 200.0)
    more_energy = model.storage_capex_usd(50.0, 400.0)
    assert more_power > base
    assert more_energy > base
    assert more_power != more_energy


def test_capex_is_linear_and_homogeneous(model):
    """No hidden economies of scale — the model is deliberately linear."""
    single = model.total_capex_usd(100.0, 50.0, 200.0)
    double = model.total_capex_usd(200.0, 100.0, 400.0)
    assert double == pytest.approx(2.0 * single, rel=1e-12)
    assert model.total_capex_usd(0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_breakdown_sums_to_the_total(model):
    parts = model.breakdown(150.0, 70.0, 500.0)
    assert parts["solar_musd"] + parts["storage_musd"] + parts["land_musd"] == pytest.approx(
        parts["total_musd"], rel=1e-12
    )


def test_metadata_states_its_scope(model):
    meta = model.metadata()
    assert "year-0 capital only" in meta["basis"]
    assert "NLR/NREL" in meta["source_ratio"]
    assert meta["storage_cost_per_kwh_derived"] > 0
    assert meta["storage_cost_per_kw_derived"] > 0


def test_zero_energy_share_would_reproduce_a_power_only_model():
    """Sanity on the parameterisation itself."""
    model = CapitalCostModel(
        solar_cost_per_kw=1000.0, storage_cost_per_kw_at_ref=1000.0, energy_share=0.0
    )
    assert model.storage_cost_per_kwh == 0.0
    assert model.storage_cost_per_kw == 1000.0
    # ... and duration would then be free, which is exactly the defect (A6).
    assert model.storage_capex_usd(10.0, 40.0) == model.storage_capex_usd(10.0, 400.0)
