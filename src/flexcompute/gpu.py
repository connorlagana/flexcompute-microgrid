"""GPU power-performance model.

The mapping from GPU power to useful compute is the single most
outcome-determining assumption in this project, so it is treated as data with
provenance rather than as a constant in the code.

Every curve carries a :class:`CurveProvenance` recording what it is, where it
came from, and — critically — *how it was derived*. Three kinds exist:

``synthetic``
    Invented for structural testing. Shape is plausible; values mean nothing.
``literature_derived``
    Built from numbers published in a real study, but with a documented
    inference step applied on top (e.g. using SM clock as a throughput proxy).
    Real provenance, not a direct measurement of the quantity we want.
``measured``
    Directly measured power-vs-throughput for the stated GPU and workload. We
    do not currently ship one.

The kind travels with every result. `DispatchResult.metadata` carries it, the
comparison report prints it, and `warn_if_not_measured` exists so a headline
number can never be quoted off a synthetic curve by accident.

Refusal to extrapolate
----------------------
A curve is defined only over the power range its source actually covers.
Asking for a power fraction below `min_operating_power_fraction` does not
silently extrapolate — it returns zero compute. Inventing performance data
below the measured floor is exactly the failure mode this module exists to
prevent.

Device curve vs fleet curve
---------------------------
A per-device curve does not describe a fleet. With thousands of GPUs the
facility can run a **mix** of per-device power states, so the set of
(power, compute) points the *fleet* can reach is the convex hull of the
per-device curve anchored at (idle, 0). :class:`GpuFleet` applies that lift by
default (``aggregation="time_shared"``); ``"per_device"`` keeps the sharp
device curve, with a hard floor below which the fleet parks entirely.

The lift buys two things: the fleet can operate continuously down to idle
without any point being an extrapolation, and the resulting compute function is
**concave**, which is what makes the MPC's planning problem a linear program
rather than a mixed-integer one. It is also optimistic — it assumes free
rack-granular control and gives partial credit during a brownout — so both
aggregations should be reported for any headline claim. See ASSUMPTIONS B8.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

CurveKind = Literal["synthetic", "literature_derived", "measured"]

#: What the x-axis of a curve actually is.
#:
#: ``measured_draw``  -- power the GPU was observed to consume.
#: ``power_cap``      -- the cap that was *configured*. A part-loaded GPU may
#:                       draw less than its cap, so treating a cap as a draw
#:                       understates real consumption at that operating point.
#:                       Curves in this basis must say so; the simulator spends
#:                       the number as if it were consumption.
PowerBasis = Literal["measured_draw", "power_cap"]

#: What the y-axis actually is.
#:
#: ``direct_measurement`` -- the source reports application throughput.
#: ``inferred``           -- throughput was derived from a proxy (SM clock,
#:                           FLOPS estimate) by us or by the source.
ThroughputBasis = Literal["direct_measurement", "inferred"]


@dataclass(frozen=True)
class CurveProvenance:
    """Where a power-performance curve came from, and what was assumed.

    Aliased as :data:`GpuPerformanceProfile`, which reads better at a call site
    that is describing a piece of hardware running a workload rather than
    annotating an array.

    ``power_basis`` and ``throughput_basis`` are separate fields because they
    fail independently. A curve can have directly measured throughput plotted
    against a configured power cap — which is the best available data for H100
    LLM training — and reporting it as "measured" without qualification would
    overstate one axis while understating the other.
    """

    name: str
    kind: CurveKind
    gpu: str
    workload: str
    precision: str | None = None
    source_title: str | None = None
    source_authors: str | None = None
    source_id: str | None = None          # arXiv id, DOI, ...
    source_url: str | None = None
    derivation: str = ""
    caveats: tuple[str, ...] = ()
    #: e.g. "single node, 4x H100, per-GPU basis". Exists so that a few-GPU
    #: bench can never be quoted as if it characterised a 10,000-GPU fleet.
    measurement_scope: str | None = None
    power_basis: PowerBasis | None = None
    throughput_basis: ThroughputBasis | None = None
    #: Free-form pointer to the exact table or figure the numbers were read from.
    source: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


#: Preferred spelling when declaring a new hardware/workload profile.
GpuPerformanceProfile = CurveProvenance


@dataclass(frozen=True)
class PowerPerformanceCurve:
    """Monotone power-fraction -> compute-fraction mapping with provenance.

    Both axes are fractions of the workload's *unconstrained* operating point:
    power fraction 1.0 is the power the fleet draws when nothing is holding it
    back, and compute fraction 1.0 is the work it does there. Neither axis is
    a fraction of TDP or of peak FLOPS.
    """

    power_fraction: np.ndarray
    compute_fraction: np.ndarray
    provenance: CurveProvenance
    idle_power_fraction: float = 0.05

    def __post_init__(self) -> None:
        p = np.asarray(self.power_fraction, dtype=float)
        c = np.asarray(self.compute_fraction, dtype=float)
        object.__setattr__(self, "power_fraction", p)
        object.__setattr__(self, "compute_fraction", c)

        if p.shape != c.shape or p.ndim != 1 or p.size < 2:
            raise ValueError("power_fraction and compute_fraction must be 1-D and equal length")
        if np.any(np.diff(p) <= 0):
            raise ValueError("power_fraction must be strictly increasing")
        if np.any(np.diff(c) < 0):
            raise ValueError("compute_fraction must be non-decreasing (more power cannot mean less work)")
        if not (0.0 < p[0] and p[-1] == 1.0):
            raise ValueError("power_fraction must lie in (0, 1] and end exactly at 1.0")
        if c[-1] != 1.0:
            raise ValueError("compute_fraction must end exactly at 1.0 (definition of the reference point)")
        if np.any(c < 0.0) or np.any(c > 1.0):
            raise ValueError("compute_fraction must lie in [0, 1]")
        if not (0.0 <= self.idle_power_fraction < p[0]):
            raise ValueError("idle_power_fraction must be non-negative and below the operating floor")

    # -- domain ------------------------------------------------------------
    @property
    def min_operating_power_fraction(self) -> float:
        """Lowest power fraction the source data actually covers."""
        return float(self.power_fraction[0])

    @property
    def is_measured(self) -> bool:
        return self.provenance.kind == "measured"

    @property
    def measures_consumed_power(self) -> bool:
        """True when the x-axis is observed draw rather than a configured cap."""
        return self.provenance.power_basis == "measured_draw"

    def warn_if_not_measured(self) -> None:
        if not self.is_measured:
            warnings.warn(
                f"GPU curve '{self.provenance.name}' is {self.provenance.kind}, "
                "not a direct measurement. Results are structurally valid but "
                "their magnitude is not a claim about real hardware.",
                stacklevel=2,
            )

    def basis_warnings(self) -> list[str]:
        """Every way in which this curve falls short of a direct measurement.

        Separate from :meth:`warn_if_not_measured` because the two axes fail
        independently, and the primary curve fails only on one of them: its
        throughput is measured, its power is a configured cap. Reporting code
        prints this list so neither half can be quietly dropped.
        """
        problems: list[str] = []
        if not self.is_measured:
            problems.append(f"curve kind is '{self.provenance.kind}', not 'measured'")
        if self.provenance.power_basis == "power_cap":
            problems.append(
                "x-axis is the configured power cap, not measured average draw; "
                "the simulator spends the cap as if it were consumption"
            )
        elif self.provenance.power_basis is None:
            problems.append("power basis is undeclared")
        if self.provenance.throughput_basis == "inferred":
            problems.append("y-axis throughput is inferred from a proxy, not measured")
        elif self.provenance.throughput_basis is None:
            problems.append("throughput basis is undeclared")
        return problems

    # -- evaluation --------------------------------------------------------
    def compute_fraction_at(self, power_fraction) -> np.ndarray:
        """Compute produced at a given power fraction, as a fraction of full.

        Values at or above 1.0 saturate at 1.0 (extra power buys no extra work).
        Values below the measured floor return 0.0 — the fleet is parked, not
        extrapolated.
        """
        x = np.asarray(power_fraction, dtype=float)
        out = np.interp(x, self.power_fraction, self.compute_fraction)
        out = np.where(x < self.min_operating_power_fraction, 0.0, out)
        return np.clip(out, 0.0, 1.0)

    def metadata(self) -> dict:
        return {
            **self.provenance.as_dict(),
            "min_operating_power_fraction": self.min_operating_power_fraction,
            "idle_power_fraction": self.idle_power_fraction,
            "points": [
                {"power_fraction": float(p), "compute_fraction": float(c)}
                for p, c in zip(self.power_fraction, self.compute_fraction)
            ],
        }

    # -- fleet aggregation -------------------------------------------------
    def concave_hull(self) -> "PowerPerformanceCurve":
        """Upper concave hull over ``[idle_power_fraction, 1.0]``.

        A fleet of many GPUs can run a *mix* of per-device power states, so the
        set of (power, compute) points a fleet can reach is the convex hull of
        the per-device curve, anchored at (idle, 0) -- every device parked. Its
        upper boundary is this concave hull.

        Two consequences, both wanted:

        * the fleet can operate continuously down to idle, because "40% power"
          is realisable as 'run some racks, park the rest';
        * the resulting compute function is concave, which makes the planning
          problem an LP rather than a mixed-integer program.

        Where the per-device curve is already concave the hull is identical to
        it, so this changes nothing in that region.
        """
        points = [(self.idle_power_fraction, 0.0)] + list(
            zip(self.power_fraction.tolist(), self.compute_fraction.tolist())
        )
        hull = _upper_concave_hull(points)
        return PowerPerformanceCurve(
            power_fraction=np.array([p for p, _ in hull]),
            compute_fraction=np.array([c for _, c in hull]),
            idle_power_fraction=0.0,   # the hull already starts at idle
            provenance=CurveProvenance(
                **{
                    **self.provenance.as_dict(),
                    "name": f"{self.provenance.name}__fleet_hull",
                    "derivation": (
                        self.provenance.derivation
                        + " Fleet aggregation: upper concave hull anchored at "
                        "(idle, 0), representing a fleet running a mix of "
                        "per-device power states."
                    ),
                }
            ),
        )


def _upper_concave_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper concave hull of points sorted by x (monotone chain)."""
    ordered = sorted(points)
    hull: list[tuple[float, float]] = []
    for point in ordered:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            x3, y3 = point
            # Drop the middle point when it sits at or below the chord: a
            # concave boundary must have strictly decreasing slopes.
            if (y2 - y1) * (x3 - x1) <= (y3 - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


# ---------------------------------------------------------------------------
# Shipped curves
# ---------------------------------------------------------------------------

SYNTHETIC_CONCAVE_V1 = PowerPerformanceCurve(
    power_fraction=np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00]),
    compute_fraction=np.array([0.68, 0.80, 0.88, 0.94, 0.98, 1.00]),
    idle_power_fraction=0.05,
    provenance=CurveProvenance(
        name="synthetic_concave_v1",
        kind="synthetic",
        gpu="generic",
        workload="generic",
        derivation=(
            "Invented placeholder. The concave shape encodes the qualitative "
            "claim that the last watts buy the least work; the specific values "
            "are not sourced from any hardware measurement."
        ),
        caveats=(
            "Not a measurement. Use only for structural testing and sensitivity "
            "bounds, never for a quoted result.",
        ),
        measurement_scope="none (invented)",
        power_basis=None,
        throughput_basis=None,
        source="none",
    ),
)


# ---------------------------------------------------------------------------
# Mayr et al. 2026 — directly measured application throughput under power caps
# ---------------------------------------------------------------------------
#
# Source: M. Mayr, S. Wind, L. Schroeder, M. Moradi, G. Hager, H. Koestler,
# G. Wellein, "AI Application Benchmarking: Power-Aware Performance Analysis
# for Vision and Language Models", arXiv:2603.16164.
#
# Read from the paper's own result tables (HTML text layer, not digitised from
# pixels). Both workloads below were measured on the same node, at the same six
# power caps, with the same protocol.
#
# What the paper does *not* report, and what therefore has to be labelled:
# there is no table of measured average GPU power draw per cap. Figure 1 plots
# energy efficiency against performance but does not annotate the values. So
# the x-axis of these curves is the **configured power cap**, not consumption.

_MAYR_CAP_W = np.array([200.0, 300.0, 400.0, 500.0, 600.0, 700.0])

_MAYR_SOURCE = dict(
    source_title=(
        "AI Application Benchmarking: Power-Aware Performance Analysis for "
        "Vision and Language Models"
    ),
    source_authors="Mayr, Wind, Schroeder, Moradi, Hager, Koestler, Wellein",
    source_id="arXiv:2603.16164",
    source_url="https://arxiv.org/abs/2603.16164",
    measurement_scope=(
        "Single node, 4x NVIDIA H100 (Lenovo SD665-N V3), reported per-GPU. "
        "NOT a cluster-scale measurement: no multi-node interconnect, no "
        "straggler or synchronisation effects, no scale-out communication."
    ),
)

_MAYR_CAP_CAVEAT = (
    "x-axis is the *configured power cap*, not measured average draw. The "
    "paper reports no per-cap average power. A GPU may draw less than its cap, "
    "so this axis is an upper bound on consumption at each point; treating it "
    "as consumption understates compute per watt at part load. See the "
    "companion curve h100_llama3_pretrain_drawaxis_sensitivity for a bound on "
    "the size of that effect."
)


def _h100_llama3_pretrain_curve() -> PowerPerformanceCurve:
    """H100 running LLaMA 3 8B continued pre-training. **The primary curve.**

    Table 4 of arXiv:2603.16164, "Pre-training / H100" row, per-GPU tokens per
    second against the configured power cap:

    | power cap | tokens/s per GPU |
    |-----------|------------------|
    | 200 W     |  9,699.3         |
    | 300 W     | 17,982.3         |
    | 400 W     | 24,853.4         |
    | 500 W     | 27,705.3         |
    | 600 W     | 29,250.0         |
    | 700 W     | 30,271.0         |

    Both axes are normalised to the 700 W row, which is the GPU's TDP and the
    workload's unconstrained operating point.

    Why this replaces the ViT curve as primary: throughput here is the
    application's own reported token rate, logged per training step through
    framework callbacks. No proxy, no inference step of ours. And the workload
    is LLM *training*, which is what the reference model's flat 24/7 load
    profile is meant to represent — the previous curve measured vision-model
    inference and stood in for training only by assumption.

    The remaining weakness is the power axis, and it is declared rather than
    papered over: see :data:`_MAYR_CAP_CAVEAT`.
    """
    tokens_per_s = np.array([9699.3, 17982.3, 24853.4, 27705.3, 29250.0, 30271.0])
    return PowerPerformanceCurve(
        power_fraction=_MAYR_CAP_W / _MAYR_CAP_W[-1],
        compute_fraction=tokens_per_s / tokens_per_s[-1],
        # Not measured by the source. H100 idle is commonly quoted near 70-100 W
        # against a 700 W TDP; held as an explicit assumption of ours.
        idle_power_fraction=0.10,
        provenance=CurveProvenance(
            name="h100_llama3_8b_pretrain_mayr2026",
            kind="measured",
            gpu="NVIDIA H100 SXM (94 GB HBM2e, 700 W TDP), P0 power state",
            workload="LLaMA 3 8B continued pre-training, LitGPT",
            precision=(
                "not stated per-workload by the source; NGC PyTorch 2.8.0 / "
                "CUDA 12.9 defaults with TF32 available"
            ),
            derivation=(
                "Read directly from Table 4 (Pre-training, H100). Power "
                "fraction = configured cap / 700 W. Compute fraction = measured "
                "tokens/s / 30,271 tokens/s at the 700 W cap. No proxy variable "
                "and no inference step introduced by this project."
            ),
            caveats=(
                _MAYR_CAP_CAVEAT,
                "Single-node, 4-GPU measurement reported per GPU. It does not "
                "characterise a 10,000-GPU fleet; cluster-scale communication "
                "and straggler effects are absent and would flatten the curve.",
                "Batch size is 'the highest that remains stable' per the "
                "paper's protocol, not a stated number, so the operating point "
                "is not exactly reproducible from the publication alone.",
                "Continued pre-training on a ~30M-token English corpus; a "
                "production run at scale differs in data pipeline and "
                "parallelism strategy.",
                "Domain floor is 0.286 of full power; below that the fleet is "
                "modelled as parked rather than extrapolated.",
            ),
            source="arXiv:2603.16164, Table 4, row 'Pre-training / H100'",
            power_basis="power_cap",
            throughput_basis="direct_measurement",
            **_MAYR_SOURCE,
        ),
    )


def _h100_vit_train_measured_curve() -> PowerPerformanceCurve:
    """H100 running ViT-L/16 *training*, directly measured. A cross-check.

    Table 3 of arXiv:2603.16164, "ViT-L/16 / H100" row, per-GPU samples per
    second against the configured power cap.

    This exists to price one specific inference step. The previous primary
    curve took SM clock as a throughput proxy for ViT-L/16 on the same GPU
    family at the same caps. Here the same benchmark's throughput is reported
    directly, so the two can be compared and the proxy's error measured rather
    than argued about. See ``tests/test_gpu_curve.py``.
    """
    samples_per_s = np.array([116.40, 214.90, 259.00, 280.05, 297.79, 305.97])
    return PowerPerformanceCurve(
        power_fraction=_MAYR_CAP_W / _MAYR_CAP_W[-1],
        compute_fraction=samples_per_s / samples_per_s[-1],
        idle_power_fraction=0.10,
        provenance=CurveProvenance(
            name="h100_vit_l16_train_mayr2026",
            kind="measured",
            gpu="NVIDIA H100 SXM (94 GB HBM2e, 700 W TDP), P0 power state",
            workload="Vision Transformer ViT-L/16 training, 224x224 inputs",
            precision="NGC PyTorch defaults; TF32 available",
            derivation=(
                "Read directly from Table 3 (ViT-L/16, H100). Power fraction = "
                "configured cap / 700 W. Compute fraction = measured samples/s "
                "/ 305.97 samples/s at the 700 W cap."
            ),
            caveats=(
                _MAYR_CAP_CAVEAT,
                "Vision-model training, not LLM training. Retained as a "
                "sensitivity case and as a direct-measurement control for the "
                "SM-clock proxy used by h100_vit_l16_inference_ujeniya2026.",
                "Single-node, 4-GPU measurement reported per GPU.",
            ),
            source="arXiv:2603.16164, Table 3, row 'ViT-L/16 / H100'",
            power_basis="power_cap",
            throughput_basis="direct_measurement",
            **_MAYR_SOURCE,
        ),
    )


def _h100_llama3_draw_axis_curve() -> PowerPerformanceCurve:
    """The primary curve's throughput, re-plotted against measured power draw.

    A sensitivity case, and the only quantitative handle available on how much
    the power-cap axis distorts the result.

    Mayr et al. report no average draw. Ujeniya et al. (arXiv:2604.11391) do,
    for an H100 at the same six caps — but for ViT-L/16, a different workload:

    | power cap | measured avg draw (GPU + memory) |
    |-----------|----------------------------------|
    | 200 W     | 199 W |
    | 300 W     | 298 W |
    | 400 W     | 395 W |
    | 500 W     | 493 W |
    | 600 W     | 591 W |
    | 700 W     | 647 W |

    Pairing those draws with the LLM-training throughputs assumes the two
    workloads track their caps identically. They do not, necessarily: a
    memory-heavier workload sits further below its cap. So this curve is
    ``literature_derived`` and exists only to bound the error, not to be
    quoted. Note the direction — the measured draw saturates 53 W below the
    700 W cap, so the true full-power point is *lower* than the cap and this
    curve is the more pessimistic of the two for a throttling strategy.
    """
    measured_w = np.array([199.0, 298.0, 395.0, 493.0, 591.0, 647.0])
    tokens_per_s = np.array([9699.3, 17982.3, 24853.4, 27705.3, 29250.0, 30271.0])
    return PowerPerformanceCurve(
        power_fraction=measured_w / measured_w[-1],
        compute_fraction=tokens_per_s / tokens_per_s[-1],
        idle_power_fraction=0.10,
        provenance=CurveProvenance(
            name="h100_llama3_pretrain_drawaxis_sensitivity",
            kind="literature_derived",
            gpu="NVIDIA H100 (700 W TDP)",
            workload="LLaMA 3 8B continued pre-training, LitGPT",
            precision="see h100_llama3_8b_pretrain_mayr2026",
            source_title=(
                "Throughput from Mayr et al. 2026 Table 4; power draw from "
                "Ujeniya et al. 2026 Fig. 9b"
            ),
            source_authors=(
                "Mayr, Wind, Schroeder, Moradi, Hager, Koestler, Wellein "
                "(throughput); Ujeniya, Eitzinger, Hager, Wellein (power)"
            ),
            source_id="arXiv:2603.16164 + arXiv:2604.11391",
            source_url="https://arxiv.org/abs/2603.16164",
            derivation=(
                "Measured LLM-training tokens/s (arXiv:2603.16164 Table 4) "
                "plotted against measured H100 average power draw at the same "
                "caps (arXiv:2604.11391 Fig. 9b). The cross-workload "
                "substitution of the power axis is ours."
            ),
            caveats=(
                "The power axis was measured while running a DIFFERENT "
                "workload (ViT-L/16) from the throughput axis. How far a GPU "
                "sits below its cap is workload-dependent, so this pairing is "
                "an approximation and the curve must not be quoted as a "
                "measurement.",
                "Use only to bound how much the power-cap axis of the primary "
                "curve distorts the headline result.",
            ),
            source="arXiv:2603.16164 Table 4 + arXiv:2604.11391 Fig. 9b",
            measurement_scope=(
                "Both sources are single-node, few-GPU benchmarks reported "
                "per GPU."
            ),
            power_basis="measured_draw",
            throughput_basis="direct_measurement",
        ),
    )


def _h100_vit_curve() -> PowerPerformanceCurve:
    """H100 running ViT-L/16 inference, derived from Ujeniya et al. 2026.

    Two series are read directly from the numeric annotations printed in that
    paper's figures (extracted from the PDF text layer, not estimated from
    pixels), for the same benchmark on the same hardware:

      Fig. 6b - average SM frequency (MHz) per power cap, H100
      Fig. 9b - average power draw (memory + GPU, W) per power cap, H100

    | power cap | mem W | GPU W | total W | SM MHz |
    |-----------|-------|-------|---------|--------|
    | 200 W     |  72   |  127  |   199   |   625  |
    | 300 W     |  91   |  207  |   298   |  1132  |
    | 400 W     | 101   |  294  |   395   |  1501  |
    | 500 W     | 109   |  384  |   493   |  1731  |
    | 600 W     | 118   |  473  |   591   |  1889  |
    | 700 W     | 120   |  527  |   647   |  1969  |

    Power fraction uses *measured average draw* (not the cap setting),
    normalised to the 700 W row. Compute fraction uses SM frequency normalised
    the same way.

    The inference step: throughput is taken as proportional to average SM
    clock. That holds to first order for a compute-bound kernel, and ViT-L/16
    at batch 256 was chosen by the authors specifically to bypass caching. It
    is nonetheless an approximation *we* introduce, which is why this curve is
    `literature_derived` and not `measured`: the paper plots samples/sec
    against power draw (Fig. 6a) but does not annotate those values.
    """
    total_w = np.array([199.0, 298.0, 395.0, 493.0, 591.0, 647.0])
    sm_mhz = np.array([625.0, 1132.0, 1501.0, 1731.0, 1889.0, 1969.0])
    return PowerPerformanceCurve(
        power_fraction=total_w / total_w[-1],
        compute_fraction=sm_mhz / sm_mhz[-1],
        # H100 idle is commonly quoted near 100 W against a 700 W TDP. Held as
        # an explicit assumption: it is not measured by the source above.
        idle_power_fraction=0.10,
        provenance=CurveProvenance(
            name="h100_vit_l16_inference_ujeniya2026",
            kind="literature_derived",
            gpu="NVIDIA H100 (94 GiB HBM2e, 700 W TDP)",
            workload="Vision Transformer ViT-L/16 inference, batch size 256",
            precision="TF32 mixed precision",
            source_title=(
                "Architectural Trade-offs in the Energy-Efficient Era: "
                "A Comparative Study of power-capping NVIDIA H100 and H200"
            ),
            source_authors="Ujeniya, Eitzinger, Hager, Wellein",
            source_id="arXiv:2604.11391v2",
            source_url="https://arxiv.org/abs/2604.11391",
            derivation=(
                "Power fraction = measured average power draw (memory + GPU, "
                "Fig. 9b) normalised to the 700 W cap row. Compute fraction = "
                "average SM frequency (Fig. 6b) normalised the same way, using "
                "SM clock as a first-order throughput proxy for a compute-bound "
                "kernel. Both series read from the figures' printed numeric "
                "annotations via the PDF text layer."
            ),
            caveats=(
                "Throughput proportional to SM clock is our approximation, not "
                "the paper's measurement.",
                "ViT-L/16 inference is not LLM training; the reference model's "
                "load profile is a training workload.",
                "Single-GPU measurement; cluster-scale effects (interconnect, "
                "synchronisation stalls, stragglers) are absent.",
                "Memory power is 36% of total at the 200 W cap, so the "
                "compute-bound assumption is weakest at the low end.",
                "Domain floor is 0.308 of full power; below that the fleet is "
                "modelled as parked rather than extrapolated.",
            ),
            source="arXiv:2604.11391v2, Fig. 6b (SM clock) + Fig. 9b (power draw)",
            measurement_scope="Single GPU. Not a cluster-scale measurement.",
            power_basis="measured_draw",
            throughput_basis="inferred",
        ),
    )


H100_VIT_L16_INFERENCE = _h100_vit_curve()
H100_LLAMA3_PRETRAIN = _h100_llama3_pretrain_curve()
H100_VIT_L16_TRAIN = _h100_vit_train_measured_curve()
H100_LLAMA3_DRAW_AXIS = _h100_llama3_draw_axis_curve()

CURVES: dict[str, PowerPerformanceCurve] = {
    c.provenance.name: c
    for c in (
        SYNTHETIC_CONCAVE_V1,
        H100_VIT_L16_INFERENCE,
        H100_LLAMA3_PRETRAIN,
        H100_VIT_L16_TRAIN,
        H100_LLAMA3_DRAW_AXIS,
    )
}

#: The primary curve. LLM training, directly measured throughput, on the GPU
#: and workload the reference model is actually meant to represent. Changed
#: from the ViT/SM-clock proxy in Milestone 7; every result predating that
#: change was computed on the old curve and is not comparable.
DEFAULT_CURVE_NAME = "h100_llama3_8b_pretrain_mayr2026"

#: Curves a headline claim should be re-run against, in reporting order.
SENSITIVITY_CURVE_NAMES = (
    "h100_llama3_8b_pretrain_mayr2026",       # primary: LLM training, measured
    "h100_llama3_pretrain_drawaxis_sensitivity",  # same, on a measured-draw axis
    "h100_vit_l16_train_mayr2026",            # different workload, measured
    "h100_vit_l16_inference_ujeniya2026",     # the old literature-derived proxy
    "synthetic_concave_v1",                   # structural sensitivity only
)


def get_curve(name: str = DEFAULT_CURVE_NAME) -> PowerPerformanceCurve:
    if name not in CURVES:
        raise ValueError(f"Unknown GPU curve '{name}'. Available: {sorted(CURVES)}")
    return CURVES[name]


def register_curve(curve: PowerPerformanceCurve) -> None:
    """Add a curve to the registry (for user-supplied measured profiles)."""
    CURVES[curve.provenance.name] = curve


# ---------------------------------------------------------------------------
# The fleet
# ---------------------------------------------------------------------------

#: How a per-device curve is lifted to a whole fleet.
#:
#: ``time_shared`` (default) -- the fleet may run a mix of per-device power
#:   states, so its achievable set is the concave hull. Physically right for
#:   10,000 GPUs, and it makes the planning problem convex.
#: ``per_device`` -- every GPU held at the same operating point, with a hard
#:   floor below which the fleet parks entirely. Conservative; a brownout that
#:   lands below the floor scores zero work.
FleetAggregation = Literal["time_shared", "per_device"]

DEFAULT_AGGREGATION: FleetAggregation = "time_shared"


@dataclass(frozen=True)
class GpuFleet:
    """Applies a curve to a fleet, and enforces what the hardware can do.

    Separated from the curve so that "what a GPU does at 60% power" (physics,
    sourced) stays distinct from "what this facility can be asked to do"
    (aggregation, ours) and "what the controller wants" (policy, ours).
    """

    curve: PowerPerformanceCurve
    total_gpus: int
    aggregation: FleetAggregation = DEFAULT_AGGREGATION

    def __post_init__(self) -> None:
        if self.aggregation not in ("time_shared", "per_device"):
            raise ValueError(f"Unknown fleet aggregation '{self.aggregation}'")
        effective = (
            self.curve.concave_hull() if self.aggregation == "time_shared" else self.curve
        )
        object.__setattr__(self, "_effective", effective)

    @property
    def effective_curve(self) -> PowerPerformanceCurve:
        """The curve the fleet actually operates on, after aggregation."""
        return self._effective  # type: ignore[attr-defined]

    @property
    def min_power_fraction(self) -> float:
        """Lowest power fraction the fleet can hold, as a fraction of demand."""
        if self.aggregation == "time_shared":
            return self.curve.idle_power_fraction
        return self.curve.min_operating_power_fraction

    @property
    def idle_power_fraction(self) -> float:
        return self.curve.idle_power_fraction

    def clamp_request(self, requested_mw: float, unconstrained_mw: float) -> float:
        """Snap a controller's request onto a physically realisable power.

        Above unconstrained demand is pointless -- there is no more work to do.
        At the bottom, ``time_shared`` clamps to idle (park some fraction of the
        fleet); ``per_device`` snaps to idle whenever the request falls below
        the curve's measured floor, because the curve refuses to extrapolate.
        """
        if unconstrained_mw <= 0.0:
            return 0.0
        target = min(max(requested_mw, 0.0), unconstrained_mw)
        floor = self.min_power_fraction * unconstrained_mw
        if target < floor:
            return self.idle_power_fraction * unconstrained_mw
        return target

    def is_parked(self, power_mw: float, unconstrained_mw: float) -> bool:
        """True when the fleet is producing no useful work."""
        if unconstrained_mw <= 0.0:
            return True
        return self.compute_fraction(power_mw, unconstrained_mw) <= 0.0

    def compute_fraction(self, power_mw: float, unconstrained_mw: float) -> float:
        if unconstrained_mw <= 0.0:
            return 0.0
        return float(self.effective_curve.compute_fraction_at(power_mw / unconstrained_mw))

    def compute_units(self, delivered_mw, unconstrained_mw):
        """Normalised compute produced.

        One compute-unit-hour is what the fleet does in one hour at its
        unconstrained operating point, so a fixed-load facility that never
        misses an hour scores exactly 8760.
        """
        delivered = np.asarray(delivered_mw, dtype=float)
        reference = np.asarray(unconstrained_mw, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(reference > 0.0, delivered / np.where(reference > 0.0, reference, 1.0), 0.0)
        return self.effective_curve.compute_fraction_at(fraction)

    def metadata(self) -> dict:
        return {
            "total_gpus": self.total_gpus,
            "aggregation": self.aggregation,
            "min_power_fraction": self.min_power_fraction,
            "curve": self.curve.metadata(),
            "effective_points": [
                {"power_fraction": float(p), "compute_fraction": float(c)}
                for p, c in zip(
                    self.effective_curve.power_fraction, self.effective_curve.compute_fraction
                )
            ],
        }
