# The experiment pipeline

`run_config.xlsx` is the interface: one row of the `Runs` sheet is one experiment, and
`runner.py` executes the enabled rows. No code has to be written to train and evaluate a
configuration; the workbook is read, never written.

## The workbook

| Sheet | Content |
|---|---|
| `Runs` | one row per run. Row 2 carries the column bands (Meta, Recognition (q), Task, Base W0, Adapters, Anchoring, EBM prior (J), Field h(x), Readout / Baselines, Recipe, MoE, one band per task), row 3 the column names, rows 4 onwards the runs. |
| `Optionen` | the allowed / preset values of every column (the dropdown lists). Numeric columns accept any typed value; the list only presets the common ones. |
| `Parameters` | one line of documentation per column. |

The column schema is read from the workbook itself: the header row of `Runs` defines the
columns and `Optionen` their value lists, so the sheet can be extended by hand.

The important columns:

- `run_id`, `enabled`, `seed` (a comma list runs the row once per seed), `notes`.
- `task`: `blob` | `routing` | `fashion` | `cifar10_resnet` (thesis E1.1, E1.2, E2/E3
  on Fashion-MNIST, E2/E3 on CIFAR-10).
- `family`: the recognition model, `mf` | `tree` | `tree_max_span` (learned Chow-Liu
  tree) | `ebm` (exact enumeration, K <= 14) | `ebm-gibbs` (Gibbs-chain backend).
- `estimator`: `sfe-loo` | `gumbel` | blank. A blank cell resolves to the family default
  (mf -> gumbel, everything else -> sfe-loo); a tree-vs-mf comparison must set it
  explicitly to the same value on both rows. The EBM families run `sfe-loo` only.
- `K_gates`, `rank`, `w0_scale`, `j_init`, `gibbs_sweeps_S`, the field network
  (`field_kind`, `field_n_hidden`, `field_bias_b2`) and the anti-collapse recipe
  (`epochs`, `lr`, `beta_max`, `anneal_frac`, `free_bits`, `wd`, `clip`, `gamma0`,
  `warm_frac`, `anchor_type`, ...), see thesis Chapter 4.5.1 and Appendix A.12/A.13.
- `baselines`: `dropout`, `allon` or `dropout+allon` train the input-agnostic gate
  controls alongside the run.
- the task bands: `blob_*`, `routing_*`, `fashion_*` (trunk, `fashion_w0_alpha`, OOD
  probe and fraction, ...) and `c10r_*` (their CIFAR-10 counterparts).

A blank cell means the documented default of the task (`defaults.py`), so a row with only
`task` and `family` filled in reproduces the documented experiment of that task. Every
non-blank cell must take effect on its task: a cell that would not is reported as a
warning at `--dry-run` time.

## Running

```
python pipeline/runner.py --list                   # the enabled rows
python pipeline/runner.py --dry-run                # validate + resolve, write config.json only
python pipeline/runner.py                          # execute the enabled rows
python pipeline/runner.py --only <run_id> ...      # these rows, enabled or not
python pipeline/runner.py --all                    # every row
python pipeline/runner.py --isolate                # one subprocess per row (memory-capped machines)
python pipeline/runner.py --csv-en                 # comma/dot CSVs instead of the de-DE ';'/','
```

Outputs, per run and seed, in `runs/<run_id>_<timestamp>/`: `config.json` (the row as
typed, normalised, resolved, plus warnings), `metrics_seed<n>.csv/.json` (every metric with
its description) and the run figures (decision maps for blob, routing matrices, gate
activation and coupling maps, reliability diagrams, recognition-tree figures). The wide
`runs/results.xlsx` accumulates one row per run and seed; re-running a `run_id` replaces
its rows.

## results.xlsx

The shipped `runs/results.xlsx` is the result library of the thesis: the rows without a
prefix were produced by this runner; the rows prefixed `liu:`, `c10r:` and `ebmq:` were
produced by the studies under `studies/` and merged into the same sheet (`liu:` =
`fashion_lenet5`, `c10r:` = `cifar10_resnet13`, `ebmq:` = `ebm_recognition` and the E1
runner). The `notes` column of every merged row names its source file.

## Files

| File | Role |
|---|---|
| `runner.py` | row selection, per-seed execution, result files |
| `config.py` | load / normalise / validate / resolve a row |
| `defaults.py` | the blank-cell defaults per task |
| `engine.py` | one function per task: compose the `TrainConfig`, train, measure |
| `metrics.py` | the metric registry (names, sections, descriptions) and the two writers |
| `plots.py` | the per-run figures |
| `model/` | the model, the training loop and the tasks (shared with `studies/`) |
