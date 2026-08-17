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


@dataclass(frozen=True)
class CurveProvenance:
    """Where a power-performance curve came from, and what was assumed."""

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

    def as_dict(self) -> dict:
        return asdict(self)


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

    def warn_if_not_measured(self) -> None:
        if not self.is_measured:
            warnings.warn(
                f"GPU curve '{self.provenance.name}' is {self.provenance.kind}, "
                "not a direct measurement. Results are structurally valid but "
                "their magnitude is not a claim about real hardware.",
                stacklevel=2,
            )

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
        ),
    )


H100_VIT_L16_INFERENCE = _h100_vit_curve()

CURVES: dict[str, PowerPerformanceCurve] = {
    c.provenance.name: c for c in (SYNTHETIC_CONCAVE_V1, H100_VIT_L16_INFERENCE)
}

DEFAULT_CURVE_NAME = "h100_vit_l16_inference_ujeniya2026"


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
