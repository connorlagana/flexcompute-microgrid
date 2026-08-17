"""GPU power-performance curve tests.

Deliberately tests *shape and provenance*, never specific compute values. The
numbers in a curve are data with a source; asserting them here would only pin
our own transcription to itself. What must hold regardless of the data is that
the curve is monotone, normalised, honest about its origin, and refuses to
extrapolate below what its source measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from flexcompute.gpu import (
    CURVES,
    DEFAULT_CURVE_NAME,
    H100_VIT_L16_INFERENCE,
    SYNTHETIC_CONCAVE_V1,
    CurveProvenance,
    GpuFleet,
    PowerPerformanceCurve,
    get_curve,
)


@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_curve_is_normalised_and_monotone(curve):
    assert curve.power_fraction[-1] == 1.0
    assert curve.compute_fraction[-1] == 1.0
    assert np.all(np.diff(curve.power_fraction) > 0)
    assert np.all(np.diff(curve.compute_fraction) >= 0)
    assert np.all((curve.compute_fraction >= 0) & (curve.compute_fraction <= 1))


@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_curve_declares_its_provenance(curve):
    p = curve.provenance
    assert p.kind in {"synthetic", "literature_derived", "measured"}
    assert p.derivation, "every curve must say how it was produced"
    if p.kind != "synthetic":
        assert p.source_id and p.source_url, "non-synthetic curves must cite a source"
        assert p.caveats, "a sourced curve must state what it does not cover"


@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_curve_refuses_to_extrapolate_below_its_domain(curve):
    floor = curve.min_operating_power_fraction
    assert curve.compute_fraction_at(floor * 0.999) == 0.0
    assert curve.compute_fraction_at(0.0) == 0.0
    assert curve.compute_fraction_at(floor) > 0.0


@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_extra_power_buys_no_extra_compute(curve):
    assert curve.compute_fraction_at(1.0) == 1.0
    assert curve.compute_fraction_at(1.5) == 1.0
    assert curve.compute_fraction_at(50.0) == 1.0


@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_interpolation_is_monotone(curve):
    xs = np.linspace(0.0, 1.2, 400)
    ys = curve.compute_fraction_at(xs)
    assert np.all(np.diff(ys) >= -1e-12)


@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_throttling_never_improves_absolute_compute(curve):
    """Less power must not produce more work. Guards against a sign error
    turning the whole experiment into an artefact."""
    xs = np.linspace(curve.min_operating_power_fraction, 1.0, 200)
    assert np.all(np.diff(curve.compute_fraction_at(xs)) >= -1e-12)


@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_curve_is_concave_so_throttling_is_energy_efficient(curve):
    """compute_fraction >= power_fraction over the measured domain.

    This is the property that makes flexible operation interesting: backing off
    costs proportionally less work than it saves power. It is an observation
    about the shipped curves, not a law -- if a future measured curve violates
    it, this test failing is the correct alarm, because the project's premise
    would then need re-examining.
    """
    assert np.all(curve.compute_fraction >= curve.power_fraction - 1e-12)


def test_default_curve_is_the_sourced_one():
    assert get_curve().provenance.name == DEFAULT_CURVE_NAME
    assert get_curve() is H100_VIT_L16_INFERENCE
    assert H100_VIT_L16_INFERENCE.provenance.kind == "literature_derived"


def test_synthetic_curve_is_flagged_as_synthetic():
    assert SYNTHETIC_CONCAVE_V1.provenance.kind == "synthetic"
    assert not SYNTHETIC_CONCAVE_V1.is_measured


def test_no_shipped_curve_claims_to_be_measured():
    """We do not currently have direct power-vs-throughput measurements.

    When one is added this test should be updated deliberately, not deleted
    accidentally.
    """
    assert not any(c.is_measured for c in CURVES.values())


def test_non_measured_curve_warns():
    with pytest.warns(UserWarning, match="not a direct measurement"):
        get_curve().warn_if_not_measured()


def test_metadata_round_trips_for_results():
    meta = get_curve().metadata()
    assert meta["kind"] == "literature_derived"
    assert meta["gpu"].startswith("NVIDIA H100")
    assert "ViT-L/16" in meta["workload"]
    assert meta["source_id"].startswith("arXiv:")
    assert len(meta["points"]) == len(get_curve().power_fraction)


# ---------------------------------------------------------------------------
# Validation rejects malformed curves
# ---------------------------------------------------------------------------

def _provenance() -> CurveProvenance:
    return CurveProvenance(name="t", kind="synthetic", gpu="x", workload="y", derivation="test")


@pytest.mark.parametrize(
    "power,compute",
    [
        ([0.5, 0.4, 1.0], [0.5, 0.6, 1.0]),   # power not increasing
        ([0.5, 0.8, 1.0], [0.9, 0.6, 1.0]),   # compute decreasing
        ([0.5, 0.8, 0.9], [0.5, 0.8, 1.0]),   # power does not reach 1.0
        ([0.5, 0.8, 1.0], [0.5, 0.8, 0.9]),   # compute does not reach 1.0
        ([0.0, 0.8, 1.0], [0.0, 0.8, 1.0]),   # zero power is not a valid point
    ],
)
def test_malformed_curves_are_rejected(power, compute):
    with pytest.raises(ValueError):
        PowerPerformanceCurve(
            power_fraction=np.array(power, dtype=float),
            compute_fraction=np.array(compute, dtype=float),
            provenance=_provenance(),
        )


def test_idle_must_sit_below_the_operating_floor():
    with pytest.raises(ValueError):
        PowerPerformanceCurve(
            power_fraction=np.array([0.5, 1.0]),
            compute_fraction=np.array([0.6, 1.0]),
            provenance=_provenance(),
            idle_power_fraction=0.7,
        )


# ---------------------------------------------------------------------------
# Fleet behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aggregation", ["per_device", "time_shared"])
def test_fleet_clamps_requests_into_the_feasible_band(aggregation):
    curve = get_curve()
    fleet = GpuFleet(curve=curve, total_gpus=10_000, aggregation=aggregation)
    demand = 10.0
    floor = fleet.min_power_fraction * demand

    assert fleet.clamp_request(20.0, demand) == demand          # no work above demand
    assert fleet.clamp_request(demand, demand) == demand
    assert fleet.clamp_request(floor, demand) == pytest.approx(floor)
    # Below the fleet's floor it parks at idle rather than extrapolating
    parked = fleet.clamp_request(floor * 0.5, demand)
    assert parked == pytest.approx(curve.idle_power_fraction * demand)
    assert fleet.is_parked(parked, demand)
    assert fleet.clamp_request(-5.0, demand) == pytest.approx(curve.idle_power_fraction * demand)


def test_fleet_handles_zero_demand():
    fleet = GpuFleet(curve=get_curve(), total_gpus=10)
    assert fleet.clamp_request(5.0, 0.0) == 0.0
    assert fleet.is_parked(0.0, 0.0)
    assert float(np.asarray(fleet.compute_units(np.array([0.0]), np.array([0.0])))[0]) == 0.0


def test_full_delivery_scores_exactly_one_compute_unit_per_hour():
    """Normalisation anchor: a fixed-load facility that never misses scores 8760."""
    fleet = GpuFleet(curve=get_curve(), total_gpus=10_000)
    demand = np.full(8760, 9.125)
    units = fleet.compute_units(demand, demand)
    assert units.sum() == pytest.approx(8760.0)


# ---------------------------------------------------------------------------
# Fleet aggregation: the concave hull
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_concave_hull_is_strictly_concave(curve):
    """Required for the MPC to be an LP rather than a MIP."""
    hull = curve.concave_hull()
    slopes = np.diff(hull.compute_fraction) / np.diff(hull.power_fraction)
    assert np.all(np.diff(slopes) < 0), f"slopes not strictly decreasing: {slopes}"


@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_concave_hull_dominates_the_per_device_curve(curve):
    """Fleet mixing can only add options, never remove them."""
    hull = curve.concave_hull()
    xs = np.linspace(curve.idle_power_fraction, 1.0, 300)
    assert np.all(hull.compute_fraction_at(xs) >= curve.compute_fraction_at(xs) - 1e-12)


@pytest.mark.parametrize("curve", list(CURVES.values()), ids=list(CURVES))
def test_concave_hull_starts_at_idle_and_ends_at_full(curve):
    hull = curve.concave_hull()
    assert hull.power_fraction[0] == pytest.approx(curve.idle_power_fraction)
    assert hull.compute_fraction[0] == 0.0
    assert hull.power_fraction[-1] == 1.0
    assert hull.compute_fraction[-1] == 1.0


def test_hull_and_per_device_agree_where_the_curve_is_already_concave():
    """The hull only changes the low-power region it had to convexify."""
    curve = get_curve()
    hull = curve.concave_hull()
    xs = np.linspace(0.47, 1.0, 200)   # above the first hull vertex
    np.testing.assert_allclose(
        hull.compute_fraction_at(xs), curve.compute_fraction_at(xs), rtol=1e-12
    )


def test_time_shared_fleet_can_operate_below_the_per_device_floor():
    """'40% power' is realisable as 'run some racks, park the rest'."""
    curve = get_curve()
    shared = GpuFleet(curve, 10_000, "time_shared")
    per_device = GpuFleet(curve, 10_000, "per_device")

    below_floor = 0.5 * curve.min_operating_power_fraction * 10.0
    assert shared.compute_fraction(below_floor, 10.0) > 0.0
    assert per_device.compute_fraction(below_floor, 10.0) == 0.0
    assert shared.min_power_fraction < per_device.min_power_fraction


def test_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError):
        GpuFleet(get_curve(), 10, "magic")
