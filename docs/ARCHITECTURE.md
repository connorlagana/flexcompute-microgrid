# Phase 0 — Architecture map of the Newkirk reference model

Everything below refers to the vendored copy in `upstream/`
(`acnewkirk/AI-Datacenter-Microgrid-Analysis`, commit `72972aa`), which is used
**unmodified**. Line references are to that commit.

---

## 1. The pipeline in one picture

```
                    ┌──────────────────────────────────────────┐
                    │ NSRDB PSM4 TMY  (nsrdb_loader.py:35)     │
                    │ temp_air, RH, ghi, dni, dhi, wind_speed  │
                    └───────────────┬──────────────────────────┘
                                    │  one 8760-row DataFrame, reused everywhere
              ┌─────────────────────┼─────────────────────────┐
              ▼                     ▼                         ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────────┐
   │ pue_tool.py      │   │ it_facil.py      │   │ pvstoragesim.py        │
   │ (T,RH) → PUE     │   │ GPUs → IT MW     │   │ pvlib ModelChain       │
   │ picks best of 6  │   │ × CSV load shape │   │ → p_dc normalised 0..1 │
   │ cooling cases    │   │                  │   │                        │
   └────────┬─────────┘   └────────┬─────────┘   └───────────┬────────────┘
            │  hourly_pue[8760]    │ hourly_it_load_mw[8760] │ p_dc[8760]
            └──────────┬───────────┘                         │
                       ▼                                     │
          ┌────────────────────────────┐                     │
          │ FacilityLoad               │                     │
          │  cooling = it × (pue − 1)  │                     │
          └────────────┬───────────────┘                     │
                       │                                     │
                       ▼                                     ▼
          ┌──────────────────────────────────────────────────────────┐
          │ pvstoragesim.evaluate_system                             │
          │   bus_load = it×m_it + cooling×m_cool                    │
          │   solar_dc = p_dc × solar_capacity_mw                    │
          │   → simulate_battery_operation  (8760-step greedy loop)  │
          └────────────┬─────────────────────────────────────────────┘
                       │ SimulationResult (uptime, curtailment, cycles, hourly)
                       ▼
          ┌──────────────────────────────────────────────────────────┐
          │ microgrid_optimizer.MicrogridOptimizer.optimize          │
          │   decision vars: (solar_mw, battery_mw)   ← 2-D only     │
          │   constraint:    uptime ≥ target in yrs 0, 13, 14, 25    │
          │   objective:     minimise CAPEX                          │
          └────────────┬─────────────────────────────────────────────┘
                       ▼
          ┌──────────────────────────────────────────────────────────┐
          │ lcoe_calc.calculate_solar_storage_lcoe                   │
          │   LCOE = NPV(capex + opex) / NPV(MWh delivered)          │
          └──────────────────────────────────────────────────────────┘
```

Every stage after the weather fetch is a pure function of 8760-element
**positional** arrays. There is no timestamp joining anywhere downstream: index
`i` is simply "hour `i` of the year" in all of them. That is convenient and
also fragile — see ASSUMPTIONS.md, "hour-of-year alignment".

---

## 2. Answers to the Phase 0 questions

### How is the 8760-hour GPU/IT load profile generated?

Three steps, all in `it_facil.py`:

1. **Scalar average power** (`it_facil.py:226-235`)
   ```
   total_nodes    = total_gpus // gpus_per_node        # 8 GPUs/node
   it_load_avg_mw = total_nodes × node_power_avg_kw / 1000    # 7.3 kW/node
   it_load_max_mw = total_nodes × node_power_max_kw / 1000    # 8.5 kW/node
   ```
   For 10,000 GPUs: 1,250 nodes → **9.125 MW average, 10.625 MW peak**.

2. **A normalised shape** read from `output_tables/hourly_load_data.csv`
   (`it_facil.py:20-61`), column `it_load_norm`. Measured properties: exactly
   8760 rows, **mean = 1.000, min = 0.937, max = 1.068**. This is a flat 24/7
   training load with ±7% jitter, not a diurnal profile.

3. **Multiplication** (`it_facil.py:266-267`):
   `hourly_it_load_mw = it_load_avg_mw × load_shape`.

The `it_load_max_mw` / `design_contingency_factor` path feeds *facility sizing*
(and hence the optimiser's search bounds); it does not feed dispatch.

> The load shape being flat is the whole reason the fixed-load architecture is
> expensive. Demand cannot move, so supply must — which is what this project
> attacks.

### Where does hourly IT load enter the solar+BESS simulation?

One place, `pvstoragesim.py:284-291`:

```python
hourly_it_load_mw      = facility_load.hourly_it_load_mw
hourly_cooling_load_mw = facility_load.hourly_cooling_load_mw
hourly_bus_load_mw = (hourly_it_load_mw * mult['bus_to_it'] +
                      hourly_cooling_load_mw * mult['bus_to_cooling'])
```

`hourly_bus_load_mw` is then a *fixed argument* to
`simulate_battery_operation` (`pvstoragesim.py:310-320`). **This is the seam
the `ComputeController` has to break**: today the load array is computed once,
before the loop starts; it must instead be produced inside the loop, one
timestep at a time, by a controller that can see the battery state.

### How are PUE and cooling load calculated?

- `pue_tool.load_pue_lookup_table` reads a `(T_oa, RH_oa) → pue` grid per
  cooling case (`pue_tool.py:43`).
- `pue_tool.calculate_annual_pue` (`pue_tool.py:70`) rounds each hour's
  temperature to 0.5 °C and humidity to 1%, merges against the grid, and fills
  misses by weighted KD-tree nearest neighbour.
- `select_optimal_cooling_system` (`pue_tool.py:138`) evaluates cases
  `{1, 2, 14, 15, 16, 17}` and picks the lowest **annual-average** PUE. For
  Dallas this selects **case 2** (large water-side economizer water chiller),
  annual PUE **1.131**.
- `FacilityLoad.__post_init__` (`it_facil.py:129-132`) then does:
  ```
  hourly_facility_load_mw = hourly_it_load_mw × hourly_pue
  hourly_cooling_load_mw  = hourly_facility_load_mw − hourly_it_load_mw
  ```

**Critical for us:** PUE depends on *weather only*. Its function signature is
literally `(weather_df, lookup_df)` — IT load is not an input. So in this model
cooling power is a strictly linear function of IT power with zero fixed
overhead. Throttle the GPUs to 50% and cooling drops to exactly 50%. Real
facilities do not behave that way. See ASSUMPTIONS.md, item A4.

### How is PV production generated?

`pvstoragesim.get_solar_generation` (`pvstoragesim.py:42-135`):

- pvlib `ModelChain` with a **single-axis tracker**, `pdc0 = 1`,
  `gamma_pdc = -0.004 /°C`, physical AOI model, no spectral losses, and a
  **lossless placeholder inverter** (`eta_inv_nom = 1`) — real conversion
  losses are applied later by `PowerFlowAnalyzer`, not here.
- The resulting AC series is **normalised by its own annual maximum**
  (`pvstoragesim.py:120-124`), giving `p_dc ∈ [0, 1]`.

So `solar_capacity_mw` means *"the DC power at the single best hour of the TMY
year"*, not STC nameplate. Measured Dallas capacity factor relative to that
peak: **0.251**.

Inverter clipping is applied afterwards in `evaluate_system`
(`pvstoragesim.py:302-308`) and differs by architecture:

| architecture | topology     | clipping treatment |
|---|---|---|
| `ac_coupled` | `lv_direct`  | PV array hard-clipped at `solar_mw / ILR` (battery is behind the inverter) |
| `ac_coupled` | `mv_coupled` | Only the solar→load path capped; DC-coupled battery recaptures the excess |
| `dc_coupled` | either       | No cap (no inverter on the PV path) |

`ILR = inverter_load_ratio = 1.2`.

### How does battery dispatch currently work?

`simulate_battery_operation` (`pvstoragesim.py:138-215`) — a greedy,
myopic, hour-by-hour loop with **no look-ahead whatsoever**:

```
for each hour t:
    if solar_at_bus[t] > bus_load[t]:            # surplus
        serve load from solar
        charge battery with whatever is left, up to min(P_rating, headroom)
        curtail the remainder
    else:                                        # deficit
        serve what solar can
        discharge up to min(P_rating, stored energy) to cover the rest
        anything still short becomes unmet_load
```

Properties worth stating explicitly:

- **Greedy, not optimal.** It always charges when it can and always discharges
  when it must. It never withholds energy, and it cannot anticipate tomorrow.
  This is the correct baseline dispatch for a fixed load — and it is also the
  thing a forecast-aware controller improves on.
- **Losses are applied symmetrically.** `battery_rte = 0.90` enters as
  `sqrt(0.90) ≈ 0.9487` on charge *and* on discharge — it appears in every
  battery path in `power_systems_estimator.py` (lines 139, 157, 179, 193, 229,
  246, 323, 332), plus converter/transformer stages. Verified: no free energy (see `tests/test_energy_conservation.py`).
- **SOC is stored energy at the battery terminal.** Charge and discharge power
  limits are enforced there, after conversion losses — not at the AC PCS
  terminal as a real inverter rating would be.
- **Charge and discharge never co-occur** in a single hour.

### How are battery MW and MWh represented?

They are **not independent**. `pvstoragesim.py:293`:

```python
battery_energy_mwh = battery_power_mw * battery_duration_hours   # default 4.0
```

and the optimiser propagates the same coupling at
`microgrid_optimizer.py:493` (`battery_mwh = b_opt * self.costs.battery_hours`).
The optimiser's decision vector is 2-D: `(solar_mw, battery_mw)`.

Compounding this, **the cost model has no energy term at all**
(`microgrid_optimizer.py:39-45`): BESS capex is `$826/kW`, quoted for an
implied 4-hour system, and `calculate_system_cost(solar_mw, battery_mw)` never
sees MWh. Decoupling duration therefore requires a *cost* change as well as a
*sizing* change — otherwise an optimiser allowed to pick duration would buy
unlimited storage energy for free. This is a hard prerequisite for
Experiment B, not a refinement.

`evaluate_system` already accepts `battery_duration_hours` as a parameter, so
the **dispatch** side needs no rewrite (verified by
`test_independent_battery_duration_is_honoured`). Only the optimiser and cost
model are hard-coded.

### How are uptime and unmet load calculated?

`pvstoragesim.py:322-341`:

```python
hours_online = np.sum(sim_results['unmet_load_mw'] < 0.001)   # < 1 kW at the bus
uptime_pct   = hours_online / total_hours * 100
```

- **Uptime is an hour count against an absolute 1 kW threshold**, not an energy
  fraction. An hour that is 0.1% short counts identically to an hour that is
  100% short.
- `energy_served_pct` is the energy-weighted companion, and is always the
  higher of the two.
- Unmet energy is measured at the *bus*, then divided by a single annual
  average bus→load multiplier to report it at the load
  (`pvstoragesim.py:329-330`) — a small approximation, since the true
  multiplier varies hour to hour with the IT/cooling mix.

**This metric is the reason the project needs a new one.** A GPU fleet
throttled to 30% draws less power, meets it easily, and scores 100% uptime
while doing a third of the work. Useful compute has to be the headline number.

### How is solar/BESS sizing optimised?

`MicrogridOptimizer.optimize` (`microgrid_optimizer.py:303-516`), two stages:

1. **Stage 1 — feasibility screening.** Latin-hypercube sampling in
   *log space* over `solar ∈ [1, 15] × design_MW`, `battery ∈ [1, 10] × design_MW`
   (`microgrid_optimizer.py:151-154`). Each sample is a year-0-only simulation;
   anything below the uptime target is discarded. Bounds expand ×1.5 if nothing
   is feasible.
2. **Stage 2 — `scipy.optimize.differential_evolution`** over the box spanned
   by the feasible Stage-1 points, `maxiter=25, popsize=10`. The objective is
   **CAPEX**, with infeasible points returning `1e9`.

Feasibility for Stage 2 means uptime ≥ target in **all four** degradation
anchor years (0, 13, 14, 25). For the Dallas baseline the binding year is
**year 25** (99.007% vs the 99.0% requirement), not year 0 — which is why the
optimised design shows 99.418% uptime when new.

Two mechanical caveats we had to work around:

- **Seeding.** `MicrogridOptimizer` accepts `seed=`, but
  `PowerSystemOptimizer.optimize_solar_storage` and
  `compare_datacenter_power_systems` never pass one, so the public entry point
  is non-deterministic. This project drives `MicrogridOptimizer` directly with
  an explicit seed.
- **Cache tolerance.** `_get_cache_key` (`microgrid_optimizer.py:158-163`)
  rounds sizings to 1 MW, so the `sim_year_0` returned with a result can be the
  simulation of a *neighbouring* design up to 0.5 MW away. Measured on the
  Dallas baseline: reported battery 122.580 MW, attached simulation 122.754 MW.
  Uptime happened to be identical, but the mismatch is real, and it is why
  `flexcompute.baseline.optimize_sizing` re-simulates the reported sizing
  before recording metrics.

### How is degradation handled?

Not year by year — by **four anchor simulations plus linear interpolation**
(`microgrid_optimizer.py:165-269`, `degradation_model.py`):

| anchor | solar factor | battery factor |
|---|---|---|
| year 0  | 1.0 | 1.0 |
| year 13 | `get_solar_capacity_factor(12)` | grey-box fade at age 12 |
| year 14 | `get_solar_capacity_factor(13)` | 1.0 (battery replaced) |
| year 25 | `get_solar_capacity_factor(24)` | replacement battery aged 12 |

- **Solar** (`degradation_model.py:147`): 1% first year, then 0.55%/yr.
- **Battery** (`degradation_model.py:410`): a grey-box surrogate — an
  Arrhenius calendar/cycling scaffold plus Gaussian-process residuals, loaded
  from `output_tables/fade_surrogate.pkl` (hence the `scikit-learn 1.8.x` pin).
  Its inputs come from `extract_battery_year_0_stats` (`degradation_model.py:292`):
  mean SOC, SOC P90, equivalent full cycles from throughput, and a mean cell
  temperature from a simple thermal model.
- Degradation is applied by **shrinking capacity in the next anchor simulation**
  (`solar_mw × factor`, `battery_mw × factor`), not by modifying dispatch.
- `interpolate_annual_energy` (`degradation_model.py:520`) linearly interpolates
  delivered energy between anchors across the 27-year evaluation.

Note that battery degradation depends on **how the battery is used**, so a
different controller changes the fade trajectory, which changes the required
sizing. That feedback loop is already present in the model and must be kept
in the eventual comparison.

### Where are the load-bearing assumptions defined?

| assumption | value | location |
|---|---|---|
| initial battery SOC | **75%**, hard-coded | `pvstoragesim.py:319` (the function default is 50, `pvstoragesim.py:147`) |
| battery duration | **4.0 h**, hard-coded | `pvstoragesim.py:271`, `microgrid_optimizer.py:36` |
| round-trip efficiency | 0.90, split √ each way | `config.py:85` |
| inverter load ratio | 1.2 | `config.py:86` |
| GPUs per node / node power | 8 / 7.3 kW avg / 8.5 kW max | `config.py:153-155` |
| design contingency | 1.05 | `config.py:156` |
| discount rate / project life | 7% / 27 yr | `config.py:230-231` |
| battery replacement | operational year 13 | `config.py:170` |
| solar degradation | 1% then 0.55%/yr | `config.py:192-193` |
| uptime threshold | 1 kW unmet at bus | `pvstoragesim.py:323` |
| LCOE denominator | MWh **delivered**, not compute | `lcoe_calc.py:310` |

---

## 3. Exact files and functions this project will touch

| File | Role now | What we do to it |
|---|---|---|
| `upstream/**` | reference model | **nothing** — vendored read-only |
| `src/flexcompute/upstream_bridge.py` | import/path plumbing | done |
| `src/flexcompute/weather.py` | TMY behind an interface | done |
| `src/flexcompute/scenario.py` | builds the shared, controller-independent `Site` | done |
| `src/flexcompute/baseline.py` | fixed-load runs + seeded sizing | done |
| `src/flexcompute/metrics.py` | metric extraction + conservation audit | done |
| `src/flexcompute/snapshot.py` | reproducibility contract | done |
| `src/flexcompute/gpu.py` | power→compute curves with provenance, fleet aggregation | done |
| `src/flexcompute/control.py` | `ComputeController` protocol + heuristics | done |
| `src/flexcompute/dispatch.py` | closed-loop simulator, cyclic SOC boundary | done |
| `src/flexcompute/forecast.py` | `SolarForecast` protocol, perfect foresight | done |
| `src/flexcompute/mpc.py` | LP planner, annual ceiling, receding-horizon MPC | done |
| `src/flexcompute/experiments.py` | Experiment A harness | done |
| `src/flexcompute/plotting.py` | figures | done |
| **cost model** (`Q2`, `A12`) | $/kWh storage term, levelised cost of compute | **next** |
| **part-load PUE** (`Q3`) | fixed + proportional cooling | **next** |

The only upstream function we *functionally* replaced is
`simulate_battery_operation`, and `flexcompute/dispatch.py` transcribes its
arithmetic step for step so that `FixedLoadController` reproduces it
bit-for-bit. Everything else — PUE, weather, PV, power-flow multipliers,
degradation, LCOE — is reused as a library, unmodified.

---

## 4. Where `ComputeController` plugs in

The current call is a one-shot vectorised setup followed by a loop over a
pre-computed array:

```python
# today, pvstoragesim.py:284-320
hourly_bus_load_mw = it_load * m_it + cooling_load * m_cool   # computed ONCE
sim = simulate_battery_operation(solar_dc, P, E, hourly_bus_load_mw, ...)
```

The minimal change is to make the load a *function of state at time t*:

```python
# proposed, flexcompute/dispatch.py
for t in range(8760):
    obs = Observation(t, soc_mwh, solar_dc_mw[t], pue[t], nameplate_it_mw[t], ...)
    it_mw = controller.choose_power(obs)          # ← the only new degree of freedom
    cooling_mw = it_mw * (pue[t] - 1.0)
    bus_load = it_mw * m_it + cooling_mw * m_cool
    ... identical dispatch arithmetic as upstream ...
    compute += gpu_curve.compute_fraction(it_mw / nameplate_it_mw[t]) * nameplate...
```

`FixedLoadController.choose_power` returns `obs.nameplate_it_mw` unconditionally,
which reduces the loop to exactly upstream's arithmetic — the invariant the
snapshot test enforces.

Three things make this safe to do incrementally:

1. `Observation` carries only *causal* information (present state, past
   history). Future solar is reachable only through a `SolarForecast` object
   that the controller must be explicitly handed — so a controller cannot
   accidentally cheat, and perfect foresight becomes an opt-in
   `PerfectSolarForecast` rather than an unnoticed leak.
2. The controller returns a *requested* power. The dispatcher decides what the
   bus can actually deliver, and records `requested` vs `delivered` separately.
   A controller cannot violate physics by asking for too much.
3. The compute accounting is a pure post-processing function of the delivered
   IT power trace, so it can be swapped without touching dispatch.

**Built as designed.** `FixedLoadController` reproduces upstream's dispatch
bit-for-bit (nine arrays, `np.array_equal`); the MPC reaches the same physics
through a separate LP whose predicted compute matches the dispatcher's measured
compute to 0.000000 compute-units.
