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
    H100_LLAMA3_DRAW_AXIS,
    H100_LLAMA3_PRETRAIN,
    H100_VIT_L16_INFERENCE,
    H100_VIT_L16_TRAIN,
    SENSITIVITY_CURVE_NAMES,
    SYNTHETIC_CONCAVE_V1,
    CurveProvenance,
    GpuFleet,
    GpuPerformanceProfile,
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


def test_default_curve_is_the_measured_llm_training_one():
    """The primary curve is LLM training with directly measured throughput.

    Changed deliberately in Milestone 7. The previous default measured vision
    inference and inferred throughput from SM clock; both the workload and the
    throughput basis were wrong for a facility modelled as running LLM
    training around the clock.
    """
    assert get_curve().provenance.name == DEFAULT_CURVE_NAME
    assert get_curve() is H100_LLAMA3_PRETRAIN
    assert "LLaMA 3 8B" in get_curve().provenance.workload
    assert "pre-training" in get_curve().provenance.workload


def test_synthetic_curve_is_flagged_as_synthetic():
    assert SYNTHETIC_CONCAVE_V1.provenance.kind == "synthetic"
    assert not SYNTHETIC_CONCAVE_V1.is_measured


def test_every_curve_declares_both_axis_bases():
    """Neither axis may be silent about what it actually is."""
    for name, curve in CURVES.items():
        if curve.provenance.kind == "synthetic":
            continue    # invented data has no basis to declare
        assert curve.provenance.power_basis in {"measured_draw", "power_cap"}, name
        assert curve.provenance.throughput_basis in {"direct_measurement", "inferred"}, name
        assert curve.provenance.source, name
        assert curve.provenance.measurement_scope, name


def test_measured_curves_have_directly_measured_throughput():
    """'measured' is a claim about the data, not a compliment.

    A curve may only carry ``kind='measured'`` if its source reports the
    workload's own throughput. Inferring throughput from a proxy makes it
    literature-derived no matter how good the proxy is.
    """
    for name, curve in CURVES.items():
        if curve.is_measured:
            assert curve.provenance.throughput_basis == "direct_measurement", name


def test_the_primary_curve_declares_its_power_cap_axis():
    """The one thing the primary curve is not is a measurement of draw.

    Mayr et al. report no per-cap average power, so the x-axis is the
    configured cap. That must be visible in the metadata that travels with
    every result, and it must be surfaced by ``basis_warnings``.
    """
    curve = get_curve()
    assert curve.provenance.power_basis == "power_cap"
    assert not curve.measures_consumed_power
    warnings_ = curve.basis_warnings()
    assert any("power cap" in w for w in warnings_), warnings_
    assert not any("throughput" in w for w in warnings_), warnings_


def test_a_fully_direct_curve_would_warn_about_nothing():
    """Guards against basis_warnings silently always returning something."""
    profile = GpuPerformanceProfile(
        name="hypothetical", kind="measured", gpu="x", workload="y",
        power_basis="measured_draw", throughput_basis="direct_measurement",
        source="nowhere", measurement_scope="test",
    )
    curve = PowerPerformanceCurve(
        power_fraction=np.array([0.5, 1.0]),
        compute_fraction=np.array([0.7, 1.0]),
        provenance=profile,
    )
    assert curve.basis_warnings() == []


def test_non_measured_curve_warns():
    with pytest.warns(UserWarning, match="not a direct measurement"):
        SYNTHETIC_CONCAVE_V1.warn_if_not_measured()
    with pytest.warns(UserWarning, match="not a direct measurement"):
        H100_VIT_L16_INFERENCE.warn_if_not_measured()


def test_metadata_round_trips_for_results():
    meta = get_curve().metadata()
    assert meta["kind"] == "measured"
    assert meta["gpu"].startswith("NVIDIA H100")
    assert "LLaMA 3 8B" in meta["workload"]
    assert meta["source_id"] == "arXiv:2603.16164"
    assert meta["power_basis"] == "power_cap"
    assert meta["throughput_basis"] == "direct_measurement"
    assert len(meta["points"]) == len(get_curve().power_fraction)


def test_measurement_scope_refuses_to_imply_fleet_scale():
    """A four-GPU bench must never read as a 10,000-GPU measurement."""
    for name in ("h100_llama3_8b_pretrain_mayr2026", "h100_vit_l16_train_mayr2026"):
        scope = CURVES[name].provenance.measurement_scope
        assert "single node" in scope.lower()
        assert "4x" in scope.lower() or "4 gpu" in scope.lower()


def test_sensitivity_set_is_complete_and_led_by_the_default():
    assert SENSITIVITY_CURVE_NAMES[0] == DEFAULT_CURVE_NAME
    assert set(SENSITIVITY_CURVE_NAMES) == set(CURVES)


# ---------------------------------------------------------------------------
# What the SM-clock proxy actually cost
# ---------------------------------------------------------------------------

def test_sm_clock_proxy_is_within_a_stated_tolerance_of_measured_vit():
    """Price the inference step the old primary curve rested on.

    ``h100_vit_l16_inference_ujeniya2026`` took SM clock as a throughput proxy.
    ``h100_vit_l16_train_mayr2026`` measures ViT-L/16 throughput directly on
    the same GPU at the same six caps. The two are not the same benchmark --
    one is inference, the other training -- so they are not expected to agree
    exactly. This pins how far apart they are, so that a future change to
    either curve has to confront the comparison rather than quietly move it.
    """
    proxy = H100_VIT_L16_INFERENCE.compute_fraction
    direct = H100_VIT_L16_TRAIN.compute_fraction
    gap = np.abs(direct - proxy)
    # Measured 2026-08: the proxy under-reads throughput at every partial cap,
    # by at most ~13 points of compute fraction.
    assert gap.max() < 0.15, f"proxy vs direct gap grew to {gap.max():.3f}"
    assert np.all(direct >= proxy - 1e-9), (
        "SM clock is expected to under-read part-load throughput, not over-read"
    )


def test_draw_axis_sensitivity_is_more_pessimistic_than_the_cap_axis():
    """The cap axis flatters throttling; the sensitivity curve bounds by how much.

    A GPU draws less than its cap at part load, so the true consumed-power
    fraction is *higher* than the cap fraction for the same throughput. Re-
    plotting against measured draw must therefore shift every partial point
    right, making part-load operation look worse, not better. If this ever
    flips, the sensitivity curve has stopped being a conservative bound.
    """
    cap, draw = H100_LLAMA3_PRETRAIN, H100_LLAMA3_DRAW_AXIS
    np.testing.assert_allclose(cap.compute_fraction, draw.compute_fraction)
    assert np.all(draw.power_fraction[:-1] > cap.power_fraction[:-1])
    xs = np.linspace(0.35, 0.95, 50)
    assert np.all(
        draw.compute_fraction_at(xs) <= cap.compute_fraction_at(xs) + 1e-12
    )


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
