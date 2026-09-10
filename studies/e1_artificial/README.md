# E1: feasibility on artificial data

The blob task (three Gaussian clouds in 2-D, thesis E1.1), the routing task (a context
block selects one of three conflicting linear rules, E1.2) and the posterior-fidelity
ladder of the recognition families (E1.3). Everything here is small enough (K <= 6) for the
true gate posterior to be enumerated exactly, so every KL divergence is exact.

## Scripts

`run_baselines.py` trains, per task and seed, the three recognition families (`mf`,
`tree`, `ebm`) and the two trained gate ablations (`dropout` at the EBM arm's own gate
rate, `allon`), evaluates them with the deployable predictor (200 prior draws, 30 Gibbs
sweeps) and measures

- accuracy, ECE, NLL, Brier score, BALD and the gate rate of every arm (Tables 6.1, 6.3);
- the routing matrix of every routing arm (Figure 6.2);
- the decision / entropy / BALD maps of the blob task (Figure 6.1);
- the exact in-run KL(q || p) of every arm, the amortised and the free refit of every
  family to every arm's frozen posterior (Table 6.4, Figure 6.4);
- the designed loopy target p*(z) ~ exp(z'Jz / 2) with J = -J0 (1 - I), fitted by every
  family and by all six non-isomorphic trees on K = 6 (Figure 6.3);
- the Gibbs-backend arm (`ebm-gibbs`) of the blob task at several sweep budgets.

`run_temp_scaling.py` retrains the arms bit-identically, fits one temperature per arm and
seed on an independent validation draw and reports the tempered metrics (the right-hand
halves of Tables 6.1 and 6.3).

```
python studies/e1_artificial/run_baselines.py --smoke            # plumbing check, ~2 min
python studies/e1_artificial/run_baselines.py                    # 5 seeds, ~1-2 h CPU
python studies/e1_artificial/run_baselines.py --seeds 5,6,7,8,9  # more seeds, rows are merged
python studies/e1_artificial/run_temp_scaling.py
```

Options: `--tasks blob,routing` reruns one task and keeps the other's rows, `--skip-ladder`
and `--skip-maps` drop the expensive parts, `--probe-only` rebuilds only the designed-target
probe. Results are written to `artifacts/e1_baselines.json` (all rows, plus the measured
Bayes ceiling and chance level), `artifacts/e1_maps.npz` (the blob maps) and
`artifacts/e1_temp.json`. Both runners are restart-safe and merge new seeds into the
existing artifact.

`lib/gap_tools.py` holds the exact posterior enumeration and the reverse-KL family fits
used by the ladder; the recognition families themselves live in `pipeline/model/`.
