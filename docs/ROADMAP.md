# Milestone plan

Each milestone is small enough to verify on its own, and each ends with a
runnable check. The ordering is chosen so that **the fixed-load invariant is
established before anything is allowed to move**, and so that no optimisation
happens before the thing being optimised is measurable.

---

## ✅ Milestone 1 — Reproducible fixed-load baseline *(complete)*

**Goal:** pin down the reference model's numbers so later changes cannot move
them silently.

Delivered:

- `upstream/` vendored read-only at commit `72972aa`; its own 51-test suite
  passes under our environment.
- Weather behind a provider interface (PVGIS default, NSRDB optional),
  normalised to local standard hour-of-year, disk-cached.
- `Scenario → Site`: the controller-independent world, built with zero upstream
  modification by injecting weather at `DatacenterAnalyzer.weather_df`.
- Seeded sizing optimisation (upstream's own entry point is unseeded).
- Energy-conservation audit on every run: PV accounting, battery state
  transition, SOC bounds, power limits, bus balance.
- Committed baseline snapshot + `--check` mode, verified bit-reproducible.
- 44 project tests; `docs/ARCHITECTURE.md`; `ASSUMPTIONS.md`.

Found along the way: the reference optimiser's 1 MW simulation cache means the
sizing it reports can differ from the sizing it simulated (A13); and the
hard-coded 75% initial SOC is worth ~0.09 pp of uptime at the binding
constraint (A7).

**Check:** `python scripts/run_baseline.py --check && python -m pytest`

---

## ✅ Milestone 2 — `ComputeController` seam *(complete)*

**Goal:** make GPU power a decision variable, and prove the change is inert
when the decision is "always full power".

Delivered:

- `gpu.py` — `PowerPerformanceCurve` with mandatory provenance
  (`synthetic` / `literature_derived` / `measured`), a registry, and a hard
  refusal to extrapolate below the source's measured floor. Two curves ship;
  the default is derived from published H100 power-capping measurements
  (ASSUMPTIONS B5). Curve name and kind travel with every result.
- `control.py` — `Observation` (scalars only, no future data), the
  `ComputeController` protocol, `FixedLoadController`, `SimpleThrottleController`.
- `dispatch.py` — closed-loop simulator, upstream's dispatch arithmetic
  transcribed step-by-step with the bus load produced inside the loop.
- The four-way power decomposition (ASSUMPTIONS B6) and curve-weighted compute
  accounting.
- `scripts/run_experiment_a.py` (superseded the Milestone 2 comparison script)
  and the difficult-window figure.

**Gate — all three green:**

1. `controller_target == unconstrained_demand` at all 8760 timesteps, exactly.
2. Nine dispatch arrays bit-identical to upstream (`np.array_equal`, not a
   tolerance), and six headline metrics equal under `==`.
3. The Milestone 1 snapshot still reproduces exactly.

**Check:** `python -m pytest && python scripts/run_baseline.py --check`

**What it showed.** The architecture works — demand reacts to stored energy,
the throttle controller eliminates all involuntary shortfall and never drains
the battery to zero. But it produces *less* compute than the fixed load
(−0.34%), and the gap widens as infrastructure shrinks. Two reasons, both worth
carrying forward: the system curtails 73% of its solar, so energy is not the
binding constraint; and `SimpleThrottleController` is myopic on SOC alone, so
it throttles 281 hours to avoid 51 hours of shortfall. See ASSUMPTIONS threat 8.

---

## ✅ Milestone 3 — Comparison machinery *(complete)*

**Goal:** make controller-vs-controller numbers mean something.

Delivered:

- **Q1 resolved.** `dispatch.simulate_cyclic` enforces a self-sustaining year by
  fixed-point iteration on start-of-year SOC. Converges in **2 iterations with a
  residual of exactly 0.0 MWh**, seed-independent. `simulate` keeps the fixed 75%
  default so the Milestone 1 snapshot still reproduces bit-for-bit.
- **Fleet aggregation** (ASSUMPTIONS B8): a fleet can run a *mix* of per-device
  power states, so its achievable set is the concave hull of the per-device
  curve. Physically right at 10,000-GPU scale, and it is what makes the planning
  problem convex.
- `experiments.py`: one place that guarantees every strategy sees identical
  weather, hardware, demand and boundary conditions.
- `scripts/run_experiment_a.py` — comparison, de-rated sweep, and a first read of
  the Experiment B question off the sweep.

Q3 (part-load PUE) was closed later, in Milestone 5b.

---

## ✅ Milestone 4 — Forecasts behind an interface *(partial)*

Delivered:

- `forecast.py` with the `SolarForecast` protocol and `PerfectSolarForecast`.
  `Observation` still carries no future data, so foresight is only reachable by
  explicitly handing a controller a forecast object — visible at the call site.
- Year-end handling is **truncation, never padding or wrapping**. Zero-padding
  invented a week-long December night and was observed driving the LP infeasible
  at long horizons; wrapping would be acausal.

Remaining: `NoisySolarForecast` (seeded, parameterised error),
`HistoricalWeatherForecast`, `ProbabilisticSolarForecast`, and the
`HandmerGovernor` heuristic.

---

## ✅ Milestone 5a — Perfect-foresight MPC *(complete)*

**Goal:** establish the theoretical ceiling before introducing forecast error.

Delivered:

- `mpc.py`: the plant reduced to an LP — concave piecewise-linear compute over
  linear battery, conversion and bus constraints. Solved exactly with HiGHS via
  cvxpy; no heuristics, no local optima.
- `AnnualPerfectForesightPlanner` — one LP over all 8760 hours with a cyclic SOC
  constraint. The true ceiling: no horizon truncation, no terminal-value
  guesswork, no free starting energy.
- `PerfectForesightMPCController` — receding horizon, re-solved hourly, executes
  only the first action. ~7 ms per solve, ~100 s per simulated year.
- An **unserved-energy slack** priced 1000x above compute, because without it the
  LP is infeasible in exactly the multi-day droughts where control matters.

**The gate that matters:** the planner is a second, independent model of the
plant, so it could be optimal against physics the simulator does not implement.
`test_annual_plan_prediction_matches_the_dispatcher` requires the LP's predicted
compute to equal the dispatcher's measured compute for the same schedule.
**Measured gap: exactly 0.000000 compute-units, at every sizing in the sweep.**

**What it showed.**

- At the fixed-load-optimal sizing even perfect foresight wins only +0.23%: 73%
  of solar is curtailed, so energy does not bind and there is nothing to win.
- De-rate the plant and the advantage grows monotonically to +9.8% at one-fifth
  the size. Read as capital: roughly **10–13% less solar+BESS for equal
  compute**, consistently across targets.
- Horizon length barely matters where it counts — 24 h of lookahead captures
  99.75% of the perfect-foresight ceiling at scale 0.40, and 48 h captures
  99.99%. That is a strong hint that a deployable controller needs a good
  day-ahead forecast rather than a great long-range one.
- `SimpleThrottleController` is worse than doing nothing at *every* sizing, and
  the gap to the MPC reaches 19.5 percentage points. Reacting to stored energy
  is not the hard part; knowing *when* to react is.
- The MPC can out-score the annual ceiling below scale 0.35 — but only by
  browning out (ASSUMPTIONS B11). Caught by the four-way accounting, not by
  luck.

---

## ✅ Milestone 5b — Experiment B *(first pass complete)*

**Goal:** the headline question — how much less plant for the same compute?

Delivered:

- **Q3 resolved** (`ASSUMPTIONS` Q3). `cooling_fixed_fraction` splits nameplate
  cooling into fixed and proportional parts, defaulting to 0.0 so every earlier
  result still reproduces. Bus load becomes affine in GPU power, so the LP
  planner gained a `beta` term; planner/simulator agreement still measures
  exactly 0.000000.
- **Q2 resolved** (`ASSUMPTIONS` Q2). `flexcompute/costs.py` prices storage as
  `$/kW × MW + $/kWh × MWh`, with the ratio taken from NLR/NREL's published
  duration table and the absolute level from upstream — so total capex at four
  hours reproduces upstream's own formula to **0.00e+00**, asserted across
  sizings and architectures.
- `minimum_capex_for_compute`: differential evolution over solar MW, battery MW
  and battery *duration*, with a feasibility repair so the reported design
  always meets the compute target.
- `scripts/run_experiment_b.py`, which decomposes the saving into the part due
  to changing the metric and the part due to flexible operation.

Found along the way: a bang-bang controller can have **no cyclic SOC fixed
point** at all, and plain iteration orbits forever (`ASSUMPTIONS` B12). Fixed by
bisecting the residual; 31 of 205 stressed configurations need that path.

---

## 🟨 Milestone 6 — Forecast error *(the controller is done; the rest is open)*

**Goal:** replace the truth with a belief and find out how much of the
perfect-foresight advantage survives.

Delivered:

- `NoisySolarForecast` — seeded, parameterised, and structurally honest in the
  one way that decides whether the experiment means anything: error scales with
  each hour's **clear-sky envelope**, not with realised output. An
  error-as-a-fraction-of-output model is identically zero during a drought, so
  it would forecast every multi-day low-solar event perfectly — precisely the
  events that size an islanded plant. Verified against real weather: of 510
  genuinely overcast daylight hours, the default forecast over-promises by more
  than 20% of clear-sky on 33 of them.
- Error grows as √(lead time), saturates at 72 h, is AR(1)-correlated on a
  24 h timescale (white noise would cancel inside a 48-hour plan and flatter
  the result), and is **exactly zero at lead 0** — so `sigma_24h = 0`
  reproduces perfect foresight bit-for-bit and the error level is a genuine
  dial between the two experiments.
- Calibration is reported, not assumed: the default level lands at **9.9%
  nRMSE of capacity at 24 h lead** on the Dallas year, and every run carries
  its realised skill in `metrics`. See ASSUMPTIONS B13, including both
  directions in which the model is biased.
- `ForecastMPCController` — the receding-horizon controller generalised to plan
  against any belief. `PerfectForesightMPCController` is now a subclass that
  **rejects any forecast that could be wrong**, so design rule 4 is mechanical
  rather than conventional: a perfect-foresight result cannot be produced under
  a noisy label, or the reverse.
- `scripts/run_forecast_sweep.py` — error level × seed × lookahead, with the
  figure in `results/forecast_sweep_*.png`.

**The gate:** `test_a_perfect_belief_reproduces_the_perfect_foresight_controller`
requires the two controllers to choose **bit-identical** actions given the same
belief. Without it, any perfect-vs-noisy gap could be an artefact of how the
two were wired rather than a measure of what the information is worth.

**What it showed.**

- At realistic day-ahead skill (9.9% nRMSE of capacity) the forecast-aware MPC
  retains **87.1% of the perfect-foresight advantage** at scale 0.40 — 8132.4
  compute-units against a ceiling of 8160.8 and a no-control 7941.2. The value
  of forecast-aware control is mostly *not* an artefact of knowing the future.
- The degradation **accelerates** with error: 96.7% retained at 3.5% nRMSE,
  87.1% at 9.9%, 59.5% at 18.1%. Current practice sits on the flat part; a
  forecast twice as bad loses roughly 40% of the prize.
- **Seed-robust.** Three error realisations at the same level land within 1.5
  compute-units of each other — 0.7 percentage points of retention — so the
  answer is a property of the error level, not of one draw.
- **The horizon result changes shape under error.** With perfect foresight
  compute is monotone in lookahead; with a belief it peaks at 48 h and turns
  down by 96 h, because past two days the forecast has decayed toward
  climatology and adds error faster than information.
- A 12-hour horizon fails badly (7788.1, below no control) — but perfect
  foresight fails there too (7802.6), so that is a horizon pathology, not a
  forecast one. Half a day cannot see across a night.
- **Reporting correction.** The earlier "% of ceiling" column flatters short
  horizons, because the ceiling is only 2.77% above doing nothing. Measured
  against the *advantage*, a 24-hour perfect forecast captures 90.8%, not
  99.75%. Retention-of-advantage is the framing used from now on.

Still open in this milestone:

- `HandmerGovernor` — the heuristic the MPC has to beat.
- Re-running **Experiment B** under forecast error: the −7.8% capital saving
  attributed to control is still a perfect-foresight number, and the sweep
  below only re-prices Experiment A's compute.
- Forecasting **PUE and demand**, not just solar (ASSUMPTIONS B9). A real
  controller mis-forecasting a hot day gets generation and cooling load wrong
  together, in the same direction.
---

## ✅ Milestone 7 — Real inputs *(complete)*

**Goal:** find out whether the result survives more realistic inputs. It did,
and it moved — in the direction that demands the most scrutiny.

Delivered:

- **`CaseyGovernor`** — the forecast-free heuristic the MPC has to beat,
  reimplemented from Casey Handmer's published description with every
  approximation recorded (B14). Structurally incapable of receiving a forecast:
  it takes `PlantConstants` (scalars only) and an ephemeris, and a test proves
  its actions are invariant to every future solar value.
- **`solar_clock.py`** — solar geometry as a *calendar*, kept rigorously apart
  from `forecast.py`, which supplies *beliefs about weather*.
- **A directly measured primary GPU curve** — H100 running LLaMA 3 8B
  pre-training, throughput logged by the training framework
  (`arXiv:2603.16164`). The first `kind="measured"` curve the project has
  shipped. Both axes now declare their basis independently, because the power
  axis is a configured cap rather than observed draw and that must not be
  hidden. Three sensitivity curves ship alongside it, including one that bounds
  the cap-vs-draw distortion.
- **Fifteen actual Dallas weather years** (2010–2024) behind a historical
  provider interface, never averaged and never mixed across sources, with an
  explicit leap-day policy and a satellite cross-check.
- **Forecast error indexed by realised nRMSE**, calibrated per year, so a sweep
  compares the same skill level in every year.
- **Experiment B, reliability-matched** (B2) alongside the equal-compute-only
  variant (B1), plus fixed-duration variants and an explicit
  cost-extrapolation flag.
- **A ~2.7× MPC speedup** from a compiled, parametrised window LP, gated by a
  test requiring agreement with the reference solver to 1e-9 on every hourly
  action.

Found along the way, and the most consequential item in this milestone: the
inherited PV model **renormalised every weather year by that year's own peak
hour**. It inflated cloudy years by up to 9.4% and reordered which year looked
sunniest — a rank-changing artefact in exactly the variable a multi-year study
measures. Corrected in `pv_model.py`, with our replication of upstream's model
chain pinned bit-for-bit by test (B16).

**Check:** `python -m pytest && python scripts/run_baseline.py --check`

---

## Milestone 8 — What the headline still needs

- **Levelised cost of *compute*** (A12) to replace the year-0 capital
  comparison: O&M, battery replacement, degradation and discounting. The
  optimised designs land at 12–15 h storage, far from where upstream's O&M and
  replacement figures were calibrated, so this is not a cosmetic addition.
- **Battery cost beyond 10 h.** The $/kW + $/kWh split is sourced over 2–10 h.
  Every free-duration optimum sits outside that range and is flagged as an
  extrapolation; a sourced long-duration cost curve would settle whether
  12–15 h storage is genuinely optimal or an artefact of extrapolating a linear
  fit.
- **Experiment B across all fifteen years**, not one representative year. Each
  forecast-MPC sizing search is hours of compute, so this needs either more
  budget or a cheaper search.
- **A second site.** Everything here is Dallas. Nothing generalises to a
  different solar climate without being re-run.
- **Cluster-scale GPU behaviour.** The primary curve is a four-GPU, single-node
  measurement. Interconnect contention and stragglers are absent and would
  flatten it.
- **Forecasting PUE and demand**, not just solar (B9). A real controller
  mis-forecasting a hot day gets generation and cooling load wrong together, in
  the same direction.
- **Satellite weather for the full window.** NSRDB PSM4 CONUS starts in 2018;
  the 15-year study runs on ERA5 reanalysis, which smooths droughts and
  therefore understates the value of control.
