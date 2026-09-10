# E5: the FLOPs-per-epoch model

A closed-form model of the floating-point operations of ONE training epoch of the
unified trainer (`pipeline/model/trainer.py`), per recognition family, validated against
XLA's own operation count of the compiled step (thesis E5, Figure 6.17, Appendix A.19).
One epoch is one full-batch step, so the cost is a single expression in N, D, C, K, r, T
and S (plus n_chains and S_q for the Gibbs-backed EBM family).

## Layout

```
lib/flops_model.py            the analytic model: 16 blocks in 5 groups, per family
experiments/measure_flops.py  captures the jitted epoch step through TrainConfig.step_out,
                              AOT-lowers it and reads XLA's cost_analysis (no epoch runs,
                              no data is needed; a laptop CPU does the grid in minutes)
reporting/collect_flops.py    model vs measurement: block attribution, per-knob slopes, kappa
reporting/fit_formula.py      least-squares refit of the monomial coefficients per backend
reporting/make_table.py       the tables of Appendix A.19
reporting/plot_flops_composition.py, plot_fashion_composition.py, plot_gibbs_scaling.py
                              the composition charts (Figure 6.17) and the growth curves
artifacts/                    flops_measurements.json and the tables / figures
```

## Conventions

- One multiply-add = 2 FLOPs. Backward passes are counted analytically per block, not as
  a factor of the forward pass.
- Three categories are kept apart: dense FLOPs, transcendental evaluations (charged with a
  calibrated constant per Gibbs site, `tau_transc`, and per estimator sample, `tau_samp`;
  GPU 104 / 25, CPU 45 / 45) and sampler operations (site updates and random draws, the
  unit a thermodynamic sampling unit would absorb).
- The five groups: sampling (the part the TSU can absorb), moments / KL, estimator,
  input dimension, other. Group membership depends on the family: under `ebm-gibbs` the
  recognition chains join the sampling group.
- XLA counts the body of a `lax.scan` once whatever its trip count, so the Gibbs sweeps
  do not register in the raw count. The harness compiles an S = 0 companion per cell and
  linearises: `body = flops(S) - flops(S=0)`, `flops = flops(S=0) + S * body`.

## Running

```
python studies/flops_model/experiments/measure_flops.py --grid smoke     # a few cells
python studies/flops_model/experiments/measure_flops.py                  # the full knob grid
python studies/flops_model/experiments/measure_flops.py --grid scaling   # K sweeps, both families
python studies/flops_model/reporting/collect_flops.py
python studies/flops_model/reporting/make_table.py
python studies/flops_model/reporting/plot_flops_composition.py
python studies/flops_model/reporting/plot_gibbs_scaling.py
```

The measurements reported in the thesis were taken on an NVIDIA L40S; on a CPU the
multiply-add layer is backend-invariant (within about 3 %) while the transcendental
constants differ, which is why they are calibrated per backend. The plotting scripts read
GPU measurements from `artifacts_cluster/` when present and fall back to the CPU
measurements otherwise.
