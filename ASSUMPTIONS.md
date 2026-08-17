# Assumptions register

Every simplification this project depends on, in one place, with its source and
its status. The rule this file exists to enforce: **a number is either measured,
cited, or labelled as an assumption.** Nothing gets to be quietly true.

Status legend:

- 🟢 **Accepted** — deliberate, documented, not planned to change.
- 🟡 **Provisional** — placeholder we intend to replace; conclusions must not
  depend on its specific value.
- 🔴 **Open** — known to threaten experiment validity; must be resolved before
  the headline result is claimed.

Last updated: 2026-08-15 (Experiment B, first pass).

---

## 0. Units and conventions

| Quantity | Unit | Note |
|---|---|---|
| Power | MW | |
| Energy | MWh | With a 1-hour timestep, power in MW over one step equals energy in MWh. Every "×1 h" is therefore invisible in the code. If the timestep ever changes, **every** such expression must be revisited. |
| Battery state of charge | MWh at the battery terminal | After conversion and round-trip losses, not at the AC bus. |
| Battery power rating | MW at the battery terminal | Real PCS ratings are AC-side; see A5. |
| Solar capacity | MW-DC | Defined as output at the single best hour of the TMY year, **not** STC nameplate (see A9). |
| Costs | 2022 USD | Inherited from the reference model. |
| Time index | local standard hour-of-year, 0…8759 | No daylight saving (see B2). |
| Compute | normalised compute-units | Deliberately not tokens or FLOPs (see B5). |

**Sign convention.** All flows are non-negative magnitudes; direction is carried
by the variable name (`battery_charge_mw`, `battery_discharge_mw`). No variable
is allowed to change meaning by changing sign.

---

## 1. Inherited from the reference model

These come from `upstream/` and are the paper's assumptions, not ours. We adopt
them so the baseline is a faithful reproduction. Where one threatens *our*
experiment specifically, that is called out.

### A1 — Hourly timestep 🟢
8760 one-hour steps. Everything sub-hourly (ramp rates, frequency response,
inverter dynamics) is invisible. Adequate for sizing energy infrastructure;
**inadequate for any claim about GPU response time or grid stability.**

### A2 — GPU fleet power model 🟢 (inherited) / 🔴 (for our purposes)
8 GPUs per node, 7.3 kW average and 8.5 kW maximum per node including
interconnect (`config.py:153-155`, sourced to Newkirk et al.). A 10,000-GPU
fleet is therefore 9.125 MW average / 10.625 MW peak IT load.

**Why this is 🔴 for us:** the reference model only ever needs *one* operating
point, because load is exogenous. We need the whole power-performance
*surface*, and this parameterisation gives us no information about behaviour
below 100%. Nothing in the reference model tells us what a GPU does at 60%
power. That curve is entirely ours to supply, and it is the single most
outcome-determining assumption in the project (see B5).

### A3 — Workload shape 🟢
`output_tables/hourly_load_data.csv`, column `it_load_norm`: 8760 values,
mean exactly 1.000, min 0.937, max 1.068 (measured). This is a **flat 24/7
training load with ±7% jitter**, not a diurnal or bursty profile.

Consequence to keep in view: the size of the prize in this project scales with
how inflexible the baseline is. A perfectly flat baseline is the most favourable
possible comparison for flexible operation. Reporting the result against a
*more* flexible baseline workload would be a fairer stress test, and is worth
doing as a sensitivity.

### A4 — PUE depends on weather only, and cooling scales linearly with IT load 🟡 *(partly addressed by Q3)*
`pue_tool.calculate_annual_pue(weather_df, lookup_df)` takes no load argument.
`FacilityLoad` then sets `cooling = it_load × (pue − 1)`.

So at 50% GPU power the model gives exactly 50% cooling power and an unchanged
PUE. Real facilities do not behave this way: chillers, pumps and fans have
fixed and near-fixed components, and **PUE rises at part load**. A model that
assumes otherwise *over-credits* throttling, because it lets the entire
facility load shrink in proportion.

This directly biases our headline result in the direction we are hoping to find.

**Partly addressed.** `cooling_fixed_fraction` (Q3) restores a non-sheddable
cooling component, which is the part that matters for throttling. What remains
inherited and unaddressed is that PUE itself is still a pure function of
weather: it does not respond to IT load, to part-load chiller COP, or to the
plant's own operating point. A proper treatment needs a load-dependent cooling
model rather than a lookup keyed on (T, RH), and that is a bigger change to the
reference model than this project has made anywhere else.

### A5 — Battery efficiency model 🟢
Round-trip efficiency 0.90 (`config.py:85`), applied as `sqrt(0.90) ≈ 0.9487`
on charge and again on discharge, plus explicit converter/transformer stages
per architecture. Verified by test: no path creates energy, and a full
charge/discharge cycle strictly loses energy.

Simplifications inside this: efficiency is **constant** — independent of
C-rate, state of charge, and temperature. Real LFP round-trip efficiency varies
by several percentage points across that space. Power limits are enforced at
the battery terminal rather than at the PCS AC terminal.

### A6 — Battery energy is `power × 4 h` 🟢 *(resolved via Q2)*
`battery_energy_mwh = battery_power_mw × battery_duration_hours`, default 4.0,
hard-coded in the optimiser (`microgrid_optimizer.py:36, 493`).

Worse, **the cost model has no energy term at all**: BESS capex is $826/kW for
an implied 4-hour system and `calculate_system_cost` never sees MWh. An
optimiser allowed to vary duration under this cost model would buy unlimited
storage energy for free.

**Resolved.** `flexcompute.costs.CapitalCostModel` prices storage as
`$/kW × MW + $/kWh × MWh`, and Experiment B searches solar MW, battery MW and
battery duration independently. Upstream's dispatch already accepted a duration
parameter, so no dispatch change was needed. Measured consequence: given the
freedom, the optimiser does **not** pick 4 hours — see Q2.

### A7 — Initial battery SOC is 75%, granted free 🟢 *(resolved via Q1)*
`evaluate_system` hard-codes `initial_soc=75.0` (`pvstoragesim.py:319`) and the
simulation is never required to return that energy at year end.

**Measured** on the Dallas baseline (PVGIS TMY, `ac_coupled`/`mv_coupled`):

| sizing | SOC₀ = 0% | SOC₀ = 75% | Δ uptime | year-end SOC | net ΔSOC at 75% |
|---|---|---|---|---|---|
| 212.5 MW / 99.2 MW | 99.075% uptime, 737.8 MWh unmet | 99.167%, 651.2 MWh | **+0.092 pp** | 206.9 MWh | **−90.7 MWh** |
| 297.5 MW / 141.7 MW | 99.806%, 160.4 MWh unmet | 99.897%, 73.8 MWh | **+0.091 pp** | 455.7 MWh | +30.6 MWh |

So the marginal design is handed ~91 MWh it never generated, and ~0.09
percentage points of uptime. In energy terms that is 0.1% of annual delivery —
negligible. But the binding constraint is *uptime at 99.0%*, and 0.09 pp is
9% of the distance from 99.0% to 99.1%. The free energy lands in January, which
is exactly when the design constraint binds.

Note the year-end SOC is **identical regardless of the starting value** (206.9
and 455.7 MWh respectively): the battery saturates during the year and forgets
its initial condition. That made the fix cheap.

**Resolved.** `dispatch.simulate_cyclic` now enforces a self-sustaining year by
fixed-point iteration. Measured: converges in **2 iterations with a residual of
exactly 0.0 MWh**, and the converged start SOC is identical from every seed
(0%, 25%, 75%, 100%). At the Dallas baseline it moves the starting state from
367.7 to 288.4 MWh; compute is unchanged and curtailment falls by 86 MWh.
`simulate` retains the fixed 75% default so the Milestone 1 snapshot still
reproduces bit-for-bit. Every controller comparison uses the cyclic mode.

### A8 — Battery degradation via four anchor years 🟢
Simulated at years 0, 13, 14, 25 with capacity multipliers applied to the
*sizing*, then linearly interpolated across 27 years. Battery fade comes from a
grey-box surrogate (Arrhenius scaffold + Gaussian-process residuals) pickled at
`output_tables/fade_surrogate.pkl`, which is why `scikit-learn` is pinned to
1.8.x. Solar: 1% first year, then 0.55%/yr.

Important for us: **fade depends on how the battery is cycled**, so a different
controller produces a different degradation trajectory and hence a different
required sizing. This feedback must be preserved in the comparison, not
short-circuited by reusing the baseline's fade factors.

### A9 — Solar production model 🟢
pvlib `ModelChain`, single-axis tracker, `pdc0=1`, `gamma_pdc=-0.004/°C`,
physical AOI model, no spectral losses, lossless placeholder inverter (real
losses applied downstream by `PowerFlowAnalyzer`). Output is normalised by its
own annual maximum.

Consequence: "212 MW of solar" means *212 MW at the best hour of the TMY year*.
It is not an STC nameplate and should not be compared to one. Measured Dallas
capacity factor on this basis: 0.251.

Not modelled: soiling, snow, availability/outages, row-to-row shading, DC
wiring mismatch, tracker backtracking losses.

### A10 — Inverter clipping 🟢
Inverter load ratio 1.2 (`config.py:86`). Clipping treatment varies by
architecture — see ARCHITECTURE.md §2. In the `ac_coupled`/`mv_coupled`
baseline only the solar→load path is capped and the DC-coupled battery
recaptures the excess.

**Known accounting gap:** in a *deficit* hour with the cap active, PV above the
cap is neither delivered, stored, nor booked as curtailment — it is counted in
`solar_generation_mwh` but appears in no sink, so the annual books do not close.
Measured at baseline sizings: **0 MWh** (does not trigger). Pinned by
`test_inverter_cap_during_deficit_strands_pv_uncounted` so that if a future
controller drives the system into that regime, we find out.

### A11 — Uptime is an hour count against a 1 kW threshold 🟢 (inherited) / 🔴 (as a metric)
`hours_online = sum(unmet_load_mw < 0.001)`. An hour short by 0.1% counts the
same as an hour short by 100%.

**This is precisely why uptime cannot be this project's headline metric.** A
throttled fleet draws less power, meets it easily, and scores 100% uptime while
doing a fraction of the work. Useful compute replaces it. Uptime is retained
only to reproduce the baseline.

### A12 — Cost model and LCOE denominator 🟡
CAPEX is used as the optimiser's objective as a proxy for LCOE. LCOE itself is
`NPV(capex + opex) / NPV(MWh delivered)` at a 7% discount rate over 27 years,
with battery replacement in operational year 13.

For this project the denominator is wrong by construction: **energy delivered
is not the product.** We will need a levelised cost of *compute*. Introducing
it is a Milestone 5 concern; until then, LCOE is recorded for baseline
comparability only.

### A13 — Optimiser cache tolerance 🟢 (worked around)
`_get_cache_key` rounds sizings to 1 MW, so the simulation returned with a
result can belong to a design up to 0.5 MW away. Measured on the Dallas
baseline: reported battery 122.580 MW, attached simulation 122.754 MW (uptime
identical, so benign here). Worked around in
`flexcompute.baseline.optimize_sizing`, which re-simulates the reported sizing
before recording anything; the discrepancy is recorded under
`optimizer_cache_drift`.

---

## 2. Introduced by this project

### B1 — Weather source defaults to PVGIS, not NSRDB 🟡
The reference model uses NSRDB PSM4 TMY via the NLR API, which requires a free
API key. To keep the project runnable and CI-testable without credentials, the
default is PVGIS TMY (`pvlib.iotools.get_pvgis_tmy`), with NSRDB available via
`--weather-source nsrdb`.

These are different datasets. Measured PVGIS Dallas TMY: 1735.7 kWh/m²/yr GHI,
2139.3 kWh/m²/yr DNI, mean air temperature 20.03 °C. Absolute MW/MWh/$ results
will not match the published paper.

**Rule:** controller-vs-controller comparisons are only valid *within* one
weather source. The source is stamped into every snapshot and every result, and
a snapshot test asserts it is declared.

### B2 — Local standard time, no daylight saving 🟢
All 8760-element arrays are indexed by local standard hour-of-year. The UTC
offset is sampled on 1 January from the site's IANA zone (Dallas: UTC−6).

Rationale: (a) it matches the NSRDB TMY convention, so switching sources stays
apples-to-apples; (b) controllers need correct hour-of-day semantics to reason
about "hours until sunrise"; (c) a DST discontinuity would corrupt positional
alignment between load, PUE and solar arrays.

Verified end-to-end: mean GHI for Dallas peaks at local hour 11–12 and is zero
at local midnight.

### B3 — Synthetic reference year 🟢
TMY months are stitched from different calendar years. We re-index them onto a
single non-leap year (2023) so the index is contiguous and hourly. This shifts
solar declination by at most a day or two relative to the source months.
Negligible for annual energy; noted for completeness. 29 February rows are
dropped.

There is also a residual **±30–60 minute labelling ambiguity** (whether a TMY
timestamp marks the beginning, middle or end of its interval) that we have not
resolved and that is consistent across all providers we use.

### B4 — Determinism 🟢
The sizing optimiser is explicitly seeded (default 20260815). Upstream's public
entry point `compare_datacenter_power_systems` does **not** seed it, which is
why this project drives `MicrogridOptimizer` directly. Weather is cached to
disk on first fetch, so every subsequent run is offline and identical. The
baseline snapshot is verified to reproduce bit-for-bit.

### B5 — GPU power-performance curve 🟡
The mapping from GPU power fraction to normalised compute fraction. Two curves
ship, each carrying its own provenance record; the kind travels with every
result. **No shipped curve is a direct measurement of power vs throughput**, and
`test_no_shipped_curve_claims_to_be_measured` enforces that we do not quietly
start pretending otherwise.

Both axes are fractions of the workload's *unconstrained* operating point — not
of TDP and not of peak FLOPS. One compute-unit-hour is what the fleet does in
one hour running unthrottled, so a fixed-load facility that never misses an
hour scores exactly 8760.

**`h100_vit_l16_inference_ujeniya2026`** — the default. `literature_derived`.

| power cap | measured draw | power fraction | SM clock | compute fraction |
|---|---|---|---|---|
| 200 W | 199 W | 0.308 | 625 MHz | 0.317 |
| 300 W | 298 W | 0.461 | 1132 MHz | 0.575 |
| 400 W | 395 W | 0.611 | 1501 MHz | 0.762 |
| 500 W | 493 W | 0.762 | 1731 MHz | 0.879 |
| 600 W | 591 W | 0.913 | 1889 MHz | 0.959 |
| 700 W | 647 W | 1.000 | 1969 MHz | 1.000 |

Source: Ujeniya, Eitzinger, Hager & Wellein, *Architectural Trade-offs in the
Energy-Efficient Era: A Comparative Study of power-capping NVIDIA H100 and
H200*, [arXiv:2604.11391v2](https://arxiv.org/abs/2604.11391). Hardware: H100
94 GiB HBM2e. Workload: ViT-L/16 inference, batch 256, TF32 mixed precision.

Both series are read from the numeric annotations printed in that paper's
figures (Fig. 6b average SM frequency, Fig. 9b power breakdown), extracted from
the PDF text layer rather than estimated from pixels. Power fraction uses
*measured average draw*, not the cap setting.

**The inference step is ours:** throughput is taken as proportional to average
SM clock. The paper plots samples/sec against power draw (Fig. 6a) but does not
annotate those values, so we substituted a first-order proxy that holds for a
compute-bound kernel. That single substitution is why this is
`literature_derived` and not `measured`. Further caveats, all recorded in the
curve's own metadata: ViT inference is not LLM training; the measurement is
single-GPU, so cluster effects (interconnect, synchronisation stalls,
stragglers) are absent; memory is 36% of package power at the 200 W cap, where
the compute-bound assumption is weakest.

**`synthetic_concave_v1`** — `synthetic`. The original invented placeholder,
retained for sensitivity bounding. Values mean nothing.

**Refusal to extrapolate.** A curve is defined only over the power range its
source covers. Below `min_operating_power_fraction` (0.308 for the H100 curve)
the fleet is modelled as *parked* — drawing idle power, producing zero compute
— rather than extrapolated. Inventing performance data below the measured floor
is the exact failure this design prevents.

**Concavity is the load-bearing property.** Compute fraction exceeds power
fraction across the whole measured domain, which is what makes backing off
cheaper in work than it is in watts. Every headline number is sensitive to it,
so: the curve is never hard-coded into a controller, results carry the curve's
name and kind, and a sensitivity sweep over curve shape is mandatory before any
figure is quoted.

`idle_power_fraction = 0.10` is a separate assumption, **not** from the source
above; H100 idle is commonly quoted near 100 W against a 700 W TDP.

### B6 — The four power quantities 🟢
Load is never a single number once a controller exists. Four are tracked
separately at every timestep, and two differences are derived from them:

| quantity | meaning |
|---|---|
| `nameplate_it_mw` | what a fixed-load facility of this size would draw (a *design* concept) |
| `unconstrained_demand_mw` | what the workload wants right now, absent energy limits (a *workload* concept) |
| `controller_target_it_mw` | what the controller asked for |
| `delivered_it_mw` | what the bus actually supplied |

```
voluntary_throttle    = unconstrained_demand - controller_target   # a choice
involuntary_shortfall = controller_target    - delivered           # a failure
```

Nameplate and unconstrained demand are currently equal, and are kept separate
anyway: deadlines, oversubscription and deferred queues will make them diverge,
and the accounting must already be ready. Collapsing the two differences into
one "unmet load" number is what would let a controller look good by simply
failing more often.

### B7 — Delivered power is derived by proportional scaling 🟢
When the bus cannot serve the requested load, delivered GPU power is scaled by
the served fraction of the bus: `delivered = target × (bus_load − unmet) / bus_load`.

Exact under the current model, because bus load is strictly proportional to IT
power (cooling is a fixed multiple of it — A4). If A4 is fixed as proposed in
Q3, this relation becomes affine rather than proportional and must be rederived.
Physically it represents a facility that browns out uniformly rather than
shedding whole racks; rack-granular shedding is not modelled.

### B8 — Fleet aggregation: the concave hull 🟡
A per-device power/compute curve does not describe a *fleet*. With 10,000 GPUs
the facility can run a **mix** of per-device power states, so the set of
(power, compute) points the fleet can reach is the convex hull of the
per-device curve anchored at (idle, 0). Its upper boundary is the concave hull,
and that is what `GpuFleet` operates on by default (`aggregation="time_shared"`;
`"per_device"` keeps the sharp per-device curve).

For the shipped H100 curve the hull **drops the 0.308 point entirely** — running
every GPU at 30.8% power is dominated by running some at 46% and parking the
rest — and is identical to the per-device curve at and above 0.461.

Two consequences, both load-bearing:

- the fleet can operate continuously down to idle, so "40% power" means
  something physical rather than an extrapolation into unmeasured territory;
- the compute function becomes **concave**, which is what makes the MPC a
  linear program rather than a mixed-integer one.

Why 🟡: it assumes rack-granular power control with no coordination cost, and it
gives partial credit for compute during an involuntary brownout (where
`per_device` would score zero). Both are optimistic. Measured effect at the
Dallas baseline: fixed-load compute rises from 8716.85 to 8717.45, i.e. +0.007%
— negligible here because brownouts are rare, but it will matter more in the
de-rated regime. Report both aggregations for any headline claim.

### B9 — What the LP planner may assume 🟡
The MPC optimises a model of the plant that is *separate code* from the
simulator, which is a real risk: an LP can be perfectly optimal against physics
the simulator does not implement. Guarded by
`test_annual_plan_prediction_matches_the_dispatcher`, which requires the
planner's predicted compute to equal the dispatcher's measured compute for the
same schedule. **Measured: gap of exactly 0.000000 compute-units.**

Modelling choices inside the LP:

- **Unserved bus energy is a priced slack**, at 1000x the compute value of a
  full-load hour. Without it the LP is simply *infeasible* whenever the plant
  cannot cover even a fully parked fleet — precisely the multi-day drought where
  control matters most. The price makes a brownout a last resort, never a trade.
- **Simultaneous charge and discharge are left unconstrained.** Round-trip
  efficiency below 1 makes doing both in one hour strictly wasteful, so the
  optimum never does it. Verified by test rather than assumed.
- **Only solar is forecast.** Future PUE and future workload demand are treated
  as known, and this remains true now that forecast error exists (B13): the
  belief replaces the solar series and nothing else. PUE is weather-driven, so
  a genuinely forecast-aware controller would face correlated uncertainty
  there too — a badly-forecast hot day is mis-predicted in *both* generation
  and cooling load, in the same direction. This understates forecast
  difficulty, and it is the largest remaining gap in the forecast-error result.

### B10 — MPC terminal value 🟡
A receding horizon has to price the energy left in the battery at the horizon
edge or it empties it every 48 hours. Stored energy is valued at what it would
yield if spent at the fleet's most compute-efficient operating point, times
`terminal_value_scale` (default 0.95). Measured for the Dallas baseline:
0.0951 compute-units per MWh.

This is an *upper* bound on the true marginal value, which makes the controller
conservative — and measurably so. At the fixed-load-optimal sizing, where 73%
of solar is curtailed and stored energy is nearly worthless at the margin, the
48-hour MPC is **worse than doing nothing** (−0.088% vs fixed load) because it
hoards energy it never needed. In the de-rated regime, where energy genuinely
binds, the same controller reaches 99.9% of the perfect-foresight ceiling.

The honest reading: `terminal_value_scale` is the only tuning knob in the
controller, a fixed scalar is a crude stand-in for a state-dependent value
function, and the annual planner — which needs no terminal value at all — is
the number to quote as the ceiling.

### B11 — The receding-horizon MPC can end the year empty 🟡
Measured, and worth stating plainly because it looks like a paradox: at
de-rated sizings (scale ≤ 0.35) the 48-hour MPC books **more** compute than the
annual perfect-foresight planner — 6980.0 vs 6979.2 at scale 0.25.

It is not a better policy. It is a brownout. The MPC's run carries **7.31 MWh of
involuntary shortfall** at every one of those sizings (the identical value is
the fingerprint of a boundary effect, not a physical one); the annual planner
carries 0.00.

Mechanism: the receding horizon truncates at 31 December, so in the final hours
the MPC has almost no lookahead, spends down the battery, and the cyclic
fixed point therefore *starts* the year empty (`soc_start = 0.00`). The LP's
lower bound on GPU power is `idle`, so the fleet still demands ~1.1 MW at the
bus through the first January night with nothing to serve it. The annual planner
avoids this because it optimises the starting state jointly with everything
else, and chooses 28.8 MWh.

Three consequences, all adopted:

- **Compute is never reported without unserved energy beside it.** A strategy
  can always buy compute by browning out, and with fleet aggregation (B8) a
  shortfall hour still books partial compute.
- The ceiling claim is stated precisely: the annual planner is the ceiling
  *among strategies that deliver what they ask for*.
- `test_exceeding_the_ceiling_always_costs_reliability` pins the invariant. If
  a strategy ever out-scores the annual planner with **no** shortfall, the LP is
  not finding the optimum and every ceiling number in the project is suspect.

A terminal SOC floor would paper over it; leaving it visible is more useful,
because a real deployment faces exactly this year-boundary problem.

### B12 — A bang-bang controller may have no cyclic fixed point 🟢
`simulate_cyclic` solves `end_soc(start_soc) = start_soc`. For a smooth
controller that map is contracting and plain iteration converges in two steps.
For `SimpleThrottleController` it is a **step function** — a hair's change in
starting SOC flips an SOC band and moves the year-end state by whole MWh — so
the iteration can orbit forever. Observed as a stable period-5 cycle at
low-solar sizings while Experiment B's optimiser was probing the corners of its
search box.

Fixed with a two-phase solve: iterate first, then bracket the residual
`end(s) - s` (non-negative at an empty battery, non-positive at a full one) and
bisect, which is robust to the jumps. A discontinuous map may have **no exact
fixed point at all**, so the bisection accepts a residual that is negligible
against annual throughput rather than demanding zero.

Measured across 205 stressed configurations: 31 needed the fallback, all
resolved by bisection, worst residual 0.22 MWh on a 2320 MWh battery — 1e-4 of
capacity, 2.5e-6 of annual energy served. The residual is recorded in
`metrics["cyclic_soc_residual_mwh"]` on every run, so it can never quietly
become free energy.

### B13 — The forecast error model 🟡
`NoisySolarForecast` is a *synthetic* error model, not a validated forecasting
system. It is the load-bearing assumption of every forecast-aware result, so
its structure is stated in full. The belief about hour `t + k`, held at `t`, is

```
truth[t+k] + envelope[t+k] * w(k) * (sigma_24h * z[t+k] + bias)
```

clipped to `[0, envelope]`, with `w(k) = (min(k, 72 h) / 24 h) ** 0.5`.

Four structural choices, and what each one is doing:

- **Error scales with the hour's clear-sky envelope, not with realised output.**
  This is the choice that decides whether the model is honest. Error expressed
  as a fraction of *realised* production is identically zero when production is
  zero — the forecast would predict every multi-day drought perfectly, and
  droughts are exactly what size an islanded plant and exactly where a
  controller can be hurt. The envelope is an empirical clear-sky proxy (the max
  at the same hour-of-day within ±15 days), so it is **climatology, not
  weather**: it encodes what a clear day in mid-March looks like at this site,
  which any operator knows from a datasheet and a calendar. Only the deviation
  from it is treated as unknown. Verified against real PVGIS weather by
  `test_the_forecast_can_miss_a_real_drought`: of 510 genuinely overcast
  daylight hours, the default forecast over-promises by more than 20% of
  clear-sky on 33 of them.
- **Error grows as the square root of lead time and saturates at 72 h.** The
  diffusive shape is the usual one; saturation says that past a few days a
  forecast has decayed to climatology and stops getting worse.
- **Error is exactly zero at lead 0.** The controller measures current
  irradiance — `Observation.solar_dc_mw` already hands every other controller
  that value, and a forecast-aware one must not be handicapped relative to
  them. Bit-exact, so `sigma_24h = 0` reproduces perfect foresight identically
  and the error level is a genuine dial between the two experiments.
- **Error is AR(1)-correlated on a 24 h timescale.** White noise would be the
  wrong shape and, worse, the *easy* shape: independent hourly errors cancel
  inside a 48-hour plan, so an MPC would barely notice them and the experiment
  would flatter itself.

**The two-sided bias, stated loudest because one half favours the hypothesis.**
The error for a given target hour keeps its sign and shape and merely *shrinks*
as that hour approaches, so successive forecasts converge monotonically on the
truth. Real forecast revisions jitter, and a controller that re-plans hourly
against a jittering belief does worse than one converging smoothly — so this is
**optimistic**. Pulling the other way, the error is fully persistent across
issue times: re-planning cannot average it away, and a badly-forecast day stays
badly forecast until it arrives — which is **pessimistic**, and is the half
that bites an energy-limited controller hardest. The two do not cancel and are
not claimed to.

**Calibration.** `sigma_24h` is a fraction of each hour's clear-sky potential,
so it is *not* directly comparable with published nRMSE. The realised
whole-year error is what results quote, and it travels in every run's metadata.
Measured on the Dallas PVGIS year, scored over daylight hours:

| `sigma_24h` | nRMSE @ 24 h lead | nMAE @ 24 h | nRMSE vs mean output |
|---|---|---|---|
| 0.05 | 3.5% of capacity | 2.6% | 7.1% |
| 0.10 | 6.8% | 5.0% | 13.8% |
| **0.15 (default)** | **9.9%** | **7.2%** | **19.9%** |
| 0.20 | 12.7% | 9.2% | 25.7% |
| 0.30 | 18.1% | 12.9% | 36.6% |

Growth with lead time at the default level: 2.2% at 1 h, 5.2% at 6 h, 7.2% at
12 h, 9.9% at 24 h, 13.4% at 48 h, 16.1% at 72 h.

The default lands near 10% of capacity at day-ahead, which is the
neighbourhood published single-site day-ahead solar forecasts occupy. That is a
**range check against general practice, not a validation against any specific
dataset** — no measured error series has been fitted, and none is claimed. The
mitigation for that weakness is the sweep: results are reported across
3.5%–18% nRMSE, so the conclusion should be read off the *shape* of the
response to error, not off the default point.

Why 🟡, in one line: the structure is defensible and its biases are declared,
but the parameters are chosen rather than fitted, and only solar is uncertain
(see B9 — future PUE and demand are still handed to the planner exactly, which
understates forecast difficulty).

---

## 3. Deliberately not modelled

Listed so that nobody has to rediscover them, and so no result over-claims.

- **C1 — GPU load transients.** Ramp rates between power states are assumed
  instantaneous at hourly resolution.
- **C2 — Millisecond-scale GPU power fluctuation.** Real AI training loads
  swing tens of MW in milliseconds at cluster scale. Invisible here, and a real
  engineering constraint on any actual implementation.
- **C3 — Electrical bus dynamics.** No frequency, voltage, inertia, protection
  coordination, or fault behaviour. Islanded operation is assumed to work.
- **C4 — Workload priority and deadlines.** All compute is currently
  interchangeable and infinitely deferrable. Real training jobs have
  checkpoints, communication barriers, and deadlines. Planned as a later
  extension.
- **C5 — Storage, networking and control-plane load** are not separated from
  IT load; they are inside the 7.3 kW/node figure.
- **C6 — Data-hall thermal inertia.** No thermal mass, so cooling responds
  instantly to IT load changes.
- **C7 — Battery augmentation.** Capacity is replaced wholesale in year 13
  rather than augmented incrementally.
- **C8 — Grid interaction.** The system is fully islanded. Curtailed solar
  earns nothing; there is no export revenue and no import backstop.
- **C9 — Degradation of power electronics** and inverter/PCS availability.

---

## 4. Open questions and proposed resolutions

### Q1 — Free energy from the initial SOC 🟢 *(resolved)*
**Problem:** A7. The year starts with a 75%-full battery that was never
charged.

**Proposal (recommended): cyclic boundary condition by fixed-point iteration.**
Simulate the year, take the year-end SOC, restart with that as the initial SOC,
repeat until `|SOC_end − SOC_start|` is below tolerance. The measurement above
shows year-end SOC is *independent of the starting value* at baseline sizings,
so this converges in a single iteration there; a general loop with a cap of
~5 iterations and a convergence check covers systems that never saturate.

Why preferred over the alternatives:
- *Warm-up period* (simulate a spin-up year, discard it) costs the same and
  leaves an arbitrary spin-up length to justify.
- *Fixing SOC₀ = 0* is conservative but penalises January unrealistically —
  a real facility commissions with a charged battery.
- *Fixing SOC₀ = SOC_end as a hard constraint* is the physically meaningful
  statement: **the year must be self-sustaining.**

Cost: one extra 8760-step simulation per evaluation (~5 ms). Negligible.

**Resolved** in `dispatch.simulate_cyclic`, exactly as proposed. Converges in 2
iterations with a residual of 0.0 MWh; the fixed point is seed-independent to
within 1e-6 MWh. Cost per evaluation: one extra 8760-step simulation (~5 ms).

### Q2 — Decoupling battery MW from MWh 🟢 *(resolved)*
**Problem:** A6, both the sizing coupling and the missing energy cost term.

**Proposal:**
1. Add `bess_cost_per_kwh` and `bess_cost_per_kw` as separate config fields,
   decomposed from the existing $826/kW 4-hour figure using a published
   power/energy split (e.g. NREL ATB or PNNL storage cost breakdowns) so the
   4-hour total still reconciles with the reference model. **Reconciliation
   must be demonstrated numerically, not asserted.**
2. Extend the optimiser's decision vector to 3-D: `(solar_mw, battery_mw,
   battery_mwh)`, with `battery_mwh ≥ battery_mw × min_duration`.
3. Keep a `--fixed-4h` mode that reproduces the current 2-D behaviour exactly,
   so the baseline snapshot stays valid.

`evaluate_system` already accepts `battery_duration_hours`, so no dispatch
rewrite is needed — verified by `test_independent_battery_duration_is_honoured`.

**Resolved as proposed.** The split uses the decomposition NLR/NREL publishes
directly:

> Total system cost ($/kW) = Battery pack cost ($/kWh) × Duration (h) + BOS cost ($/kW)

Their total installed cost across durations (2/4/6/8/10 h = 403/574/744/915/1086
$/kW) fits that form exactly with slope **$85.5/kWh** and intercept
**$232/kW** — reproducing all five published rows to within a dollar, which
`test_published_duration_table_is_reproduced_by_the_affine_fit` asserts rather
than assumes.

Only the **ratio** is borrowed. Absolute levels stay upstream's, so upstream's
4-hour figure of $1021.62/kW splits into **$412.92/kW + $152.18/kWh**, and total
capex at 4 hours reproduces upstream's own formula to **0.00e+00** — asserted
across three sizings and both architectures.

Source: NLR/NREL, *Cost Projections for Utility-Scale Battery Storage: 2023
Update*, <https://docs.nlr.gov/docs/fy23osti/85332.pdf>.

Scope limit: **year-0 capital only.** No O&M, no battery replacement, no
discounting, no degradation. Appropriate for Experiment B's capital question,
where the omitted terms scale similarly across strategies — but it is not an
LCOE and must never be quoted as one. A levelised cost of *compute* (A12)
remains open.

### Q3 — PUE at part load 🟢 *(resolved)*
**Problem:** A4. Linear cooling scaling over-credits throttling.

**Proposal:** model facility overhead as fixed + proportional:

```
cooling_mw(t) = fixed_fraction × cooling_mw_nameplate(t)
              + (1 − fixed_fraction) × cooling_mw_nameplate(t) × (it_mw / it_nameplate_mw)
```

with `fixed_fraction` configurable and defaulted conservatively (i.e.
*against* our hypothesis). At `fixed_fraction = 0` this reduces exactly to
upstream's behaviour, preserving the baseline. Report the headline result
across a sweep of `fixed_fraction`.

**Resolved as proposed**, with `cooling_fixed_fraction` defaulting to 0.0 so
every earlier result still reproduces bit-for-bit. Bus load becomes *affine* in
GPU power rather than proportional, so the planner gained a `beta` term; the
planner/simulator agreement still measures exactly 0.000000.

Measured effect on the perfect-foresight advantage over a fixed load:

| infrastructure scale | f = 0.0 | f = 0.2 | f = 0.4 |
|---|---|---|---|
| 1.00 | +0.23% | +0.23% | +0.22% |
| 0.60 | +1.28% | +1.23% | +1.17% |
| 0.40 | +2.77% | +2.58% | +2.37% |
| 0.25 | +8.65% | +7.92% | +7.14% |

So the linear-cooling assumption was inflating the benefit by roughly **15–17%
in relative terms** — a real correction, but not one that changes the
conclusion. It is smaller than expected for a specific reason worth carrying:
Dallas selects a water-side economizer with annual PUE **1.131**, so cooling is
only ~13% of facility power and even 40% of it being non-sheddable is ~5% of the
total. **At a hotter or less efficient site (PUE 1.4+) this correction would be
roughly three times larger**, and any multi-site result must sweep it.

A second, subtler effect the model now captures: fixed cooling is served *first*
during a brownout, so what reaches the GPUs falls faster than proportionally.
Even the fixed load — which never throttles — loses compute when part of its
cooling cannot be shed (`test_fixed_cooling_makes_a_brownout_hurt_the_gpus_more`).

Still an assumption, hence 🟡 not 🟢 on the *value*: the plausible range 0.2–0.4
is derived from an industry rule of thumb (a facility designed for PUE 1.4 at
full load running at PUE 1.7–2.0 at 20% load), not from a measured part-load
curve for this cooling architecture. Headline numbers are reported across the
sweep, never at a single f.

### Q4 — What counts as "useful compute"? 🟡 *(interim answer, Milestone 2)*
**Adopted:** curve-weighted, normalised so that one compute-unit-hour is what
the fleet does in one unthrottled hour. A fixed-load facility that never misses
scores 8760. A compute-unit is a **model output, not a physical measurement**,
and its magnitude inherits every caveat of B5. Still open: job-completion
semantics, which need the deadline model (C4).

Original framing, retained:
Open definitional question, not just an implementation detail. Candidates:

- **Energy-proportional:** compute ∝ delivered GPU energy. Simplest; assumes
  perfect linear scaling and makes throttling look free.
- **Curve-weighted (planned):** `Σ_t compute_fraction(power_fraction_t) ×
  nameplate_compute_per_hour`. Captures diminishing returns.
- **Job-completion:** requires a job model with deadlines (C4).

We will start with curve-weighted and state clearly that a compute-unit is a
model output, not a physical measurement.

### Q5 — What does "unmet load" mean once load is controllable? 🟢 *(resolved, Milestone 2)*
Resolved as **B6**: four quantities tracked separately, two differences derived,
voluntary throttling never conflated with involuntary shortfall. Enforced by
`test_four_way_power_decomposition` and
`test_shortfall_only_occurs_when_the_bus_is_short`.

The first measurement confirms the decomposition earns its keep. At the
fixed-load-optimal Dallas sizing, `simple_throttle` books 959 MWh of voluntary
throttling and **zero** involuntary shortfall, against `fixed_load`'s zero
voluntary and 396 MWh involuntary. A single "unmet load" number would have
scored those two as similar; they are opposites.

---

## 5. Threats to validity — running list

Kept explicit so the final write-up cannot quietly skip them.

1. **A4 (linear cooling)** biases results in favour of our hypothesis.
2. **B5 (GPU curve)** determines the magnitude of the headline result. The
   default is now sourced rather than invented, but the throughput-proportional-
   to-SM-clock step is ours, and the workload (ViT inference) is not the
   workload being modelled (LLM training).
3. **A3 (flat baseline workload)** is the most favourable possible comparison.
4. ~~**A7 (free initial energy)**~~ — resolved by the cyclic boundary condition
   (Q1). Every comparison now runs a self-sustaining year.
5. **B1 (weather source)** means absolute figures are not comparable to the
   published paper.
6. **Single TMY year, single location.** A TMY is a synthetic average year and
   under-represents multi-day low-solar events, which are exactly the events
   that size an islanded system. Multi-year and multi-site runs are needed
   before generalising.
7. **Perfect-foresight MPC is an upper bound, not an achievable design.** It
   must always be reported as such, alongside the forecast-aware result.
8. **Experiment A is the wrong place to look for the prize.** At the
   fixed-load-optimal sizing the system curtails 73% of its solar, so energy is
   not the binding constraint and even *perfect foresight* wins only +0.23%.
   Sizing is driven by the 99%-uptime tail, and compute is far more forgiving
   than uptime. The advantage grows to +9.8% at one-fifth the plant, so the
   value of flexible operation shows up as infrastructure avoided, not compute
   gained. Any framing that implies otherwise is misleading.
9. **B8 (fleet aggregation)** gives partial compute credit during brownouts and
   assumes free rack-granular power control. Both favour the hypothesis. Report
   `per_device` alongside.
10. **B10 (MPC terminal value)** is a scalar stand-in for a state-dependent
    value function, and it demonstrably mis-prices energy when curtailment is
    high. Quote the annual planner as the ceiling, not the receding-horizon MPC.
11. **The planner and the simulator are separate models.** They currently agree
    to 0.000000 compute-units, and that agreement must be re-checked after any
    change to either. It is the single point where a subtle physics
    inconsistency could invalidate every MPC number without any test failing.
12. **Most of Experiment B's saving is not attributable to control.** Measured:
    −22.4% comes from changing the success metric from 99% uptime to delivered
    compute, and only −7.8% from flexible operation on top. Reporting the −28.5%
    total as the value of forecast-aware control overstates it roughly threefold.
13. **Experiment B is not reliability-matched.** At equal compute the re-sized
    fixed-load plant still books ~395 MWh/yr of involuntary shortfall while the
    flexible designs book zero. That runs *in the flexible design's favour*, so
    the capital comparison understates its advantage — but it is not
    like-for-like, and a reliability-constrained re-run is needed before the
    number is quoted.
14. **The forecast error model is synthetic (B13).** Its structure is defensible
    and its two biases are declared — successive forecasts converge
    monotonically on the truth (optimistic), and error never averages away
    across re-plans (pessimistic) — but the parameters are chosen to land near
    published day-ahead skill, not fitted to a measured error series. Read the
    forecast-aware result off the *shape* of the response across the sweep,
    not off the default error level. And only solar is uncertain: a real
    controller mis-forecasting a hot day gets generation *and* cooling load
    wrong together, in the same direction (B9).
15. **Experiment B prices year-0 capital only.** No O&M, no battery replacement,
    no degradation, no discounting. Since the optimised designs choose ~14-hour
    storage rather than 4-hour, they sit far from the point where upstream's
    O&M and replacement figures were calibrated, and those terms may not scale
    the way this comparison implicitly assumes.
