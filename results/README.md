# Results

Two generations of results live here. **They are not comparable**, and the older
set is kept rather than deleted so the change is auditable rather than invisible.

## Current — Milestone 7 basis

Fifteen actual Dallas weather years (2010–2024, ERA5), the directly measured
H100 LLaMA 3 pre-training curve, and the corrected PV scale (`ASSUMPTIONS` B16).

| file | what it is |
|---|---|
| `experiment_a_multiyear_*_cf0.00_*.json` | six policies × 15 years × 5 scales |
| `experiment_a_multiyear_*_cf0.30_*.json` | the same with 30% of cooling non-sheddable |
| `experiment_a_multiyear_*_cf0.50_*.json` | the same with 50% non-sheddable |
| `forecast_sweep_multiyear_*.json` | compute vs realised day-ahead nRMSE, 15 years |
| `horizon_sweep_multiyear_*.json` | how far ahead a controller needs to see |
| `experiment_b2_*.json` | minimum year-0 CAPEX, B1 and B2, four duration cases |
| `fig1_controller_value_vs_scarcity.png` | controller value vs infrastructure scarcity |
| `fig2_hardest_drought.png` | the hardest solar drought in the record, four policies |
| `fig3_forecast_error_sensitivity.png` | what a wrong forecast costs |
| `fig4_economics_b1.png`, `fig4_economics_b2.png` | capital at equal compute / equal reliability |
| `fig5_year_distribution.png` | the advantage, year by year |

Every JSON records the curve name and kind, the curve's outstanding basis
caveats, the weather source and year set, the cooling fraction and the
aggregation, so a figure can always be traced to the inputs that produced it.

## Superseded — pre-Milestone-7 basis

One PVGIS TMY year, the ViT-L/16 inference curve with throughput inferred from
SM clock, and **the uncorrected per-year PV normalisation**.

```
derate_sweep_dallas_*.json / .png
experiment_a_dallas_*.json
experiment_b_dallas_*_cf0.00.json / _cf0.30.json
forecast_sweep_dallas_*.json / .png
horizon_sweep_dallas_*.json
difficult_window_dallas_*.png
```

Do not quote these. Two of the three input changes moved the answer materially:

- the LLM-training curve loses less throughput per watt given up than the ViT
  curve did, so throttling is cheaper and every controller advantage is larger;
- real weather years contain deeper multi-day droughts than a TMY, which a TMY
  construction algorithm removes by design.

The third — the PV normalisation defect — did not affect these single-TMY files
(the TMY's divisor is exactly 1.0), but it would have invalidated any multi-year
result computed on that code.
